#!/usr/bin/env python3
"""
VoiceInk — macOS menu bar push-to-talk with Qwen3-ASR on Apple Silicon.

Features:
  - Hold right Option:        push-to-talk
  - Double-tap right Option:  toggle continuous recording
  - Custom dictionary for proper noun correction
  - Screen context OCR (Vision framework) for smarter transcription
"""

import enum
import json
import logging
import logging.handlers
import os
import re
import subprocess
import threading
import time
import warnings
from pathlib import Path

# Suppress multiprocessing resource_tracker warnings about leaked semaphores.
# This is a known issue with MLX/multiprocessing — the semaphores are cleaned
# up by the OS on process exit, but the tracker warns unnecessarily.
warnings.filterwarnings("ignore", message="resource_tracker.*semaphore", category=UserWarning)

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard

try:
    import Quartz
    import Vision
    from AppKit import (
        NSObject,
        NSWorkspace,
        NSWorkspaceWillSleepNotification,
        NSWorkspaceDidWakeNotification,
        NSWorkspaceDidActivateApplicationNotification,
    )

    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False

# ── Configuration ─────────────────────────────────────────────────

# [P5-3] Defaults — overridable via settings.json
_DEFAULTS = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "sample_rate": 16000,
    "double_click_window": 0.35,
    "max_recording_secs": 1800,
    "hotkey": "alt_r",
    "ocr_languages": ["en", "zh-Hans", "zh-Hant"],
    "symbol": "waveform",
}
MODEL = _DEFAULTS["model"]
SAMPLE_RATE = _DEFAULTS["sample_rate"]
DOUBLE_CLICK_WINDOW = _DEFAULTS["double_click_window"]
MAX_RECORDING_SECS = _DEFAULTS["max_recording_secs"]
DEFAULT_HOTKEY = _DEFAULTS["hotkey"]

CONFIG_DIR = Path.home() / ".local" / "voice-input"
DICT_PATH = CONFIG_DIR / "dictionary.json"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LOG_PATH = CONFIG_DIR / "voice_input.log"

# [P2-4] Log rotation: 5MB max, 3 backups
_log_handler = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=5 * 1024 * 1024, backupCount=3
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log = logging.getLogger("voiceinput")
log.addHandler(_log_handler)
log.setLevel(logging.INFO)
try:
    os.chmod(str(LOG_PATH), 0o600)
except OSError:
    pass

# Suppress stdout/stderr to prevent launchd from mixing raw output
# into the structured log file. All logging goes through RotatingFileHandler.
import sys as _sys
if not _sys.stdout.isatty():
    _sys.stdout = open(os.devnull, 'w')
    _sys.stderr = open(os.devnull, 'w')

# Route unhandled thread exceptions to the log (stderr is now /dev/null)
import threading as _threading

def _thread_excepthook(args):
    log.error("Unhandled exception in thread %s: %s",
              args.thread.name if args.thread else "?", args.exc_value,
              exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

_threading.excepthook = _thread_excepthook


# ── State ─────────────────────────────────────────────────────────


class State(enum.Enum):
    """Application state machine.

    State transition table:
        LOADING              -> IDLE                 (models loaded)
        IDLE                 -> RECORDING_HOLD       (hotkey pressed)
        IDLE                 -> WAITING_DOUBLE_CLICK  (hotkey released quickly)
        RECORDING_HOLD       -> PROCESSING           (hotkey released)
        RECORDING_HOLD       -> IDLE                 (escape pressed / cancel)
        WAITING_DOUBLE_CLICK -> RECORDING_TOGGLE     (hotkey pressed within window)
        WAITING_DOUBLE_CLICK -> PROCESSING           (timeout, short audio recorded)
        WAITING_DOUBLE_CLICK -> IDLE                 (timeout, no audio)
        RECORDING_TOGGLE     -> PROCESSING           (hotkey pressed to stop)
        RECORDING_TOGGLE     -> IDLE                 (escape pressed / cancel)
        PROCESSING           -> IDLE                 (transcription complete)
        PROCESSING           -> ERROR                (transcription failed)
        ERROR                -> IDLE                 (after timeout / recovery)
    """

    LOADING = "loading"
    IDLE = "idle"
    RECORDING_HOLD = "recording_hold"
    WAITING_DOUBLE_CLICK = "waiting_double_click"
    RECORDING_TOGGLE = "recording_toggle"
    PROCESSING = "processing"
    ERROR = "error"  # [AUDIT-13] Visual error state

    @property
    def visual(self):
        """Map state to visual effect name for the menu bar icon."""
        if self in (State.RECORDING_HOLD, State.WAITING_DOUBLE_CLICK, State.RECORDING_TOGGLE):
            return "recording"
        if self == State.PROCESSING:
            return "processing"
        if self == State.IDLE:
            return "idle"
        if self == State.ERROR:
            return "error"
        return "loading"


# ── Sleep/Wake observer ─────────────────────────────────────────


class _SleepWakeObserver(NSObject):
    """PyObjC observer for macOS sleep/wake notifications."""
    _app = None

    def handleWillSleep_(self, notification):
        if self._app:
            self._app._on_will_sleep()

    def handleDidWake_(self, notification):
        if self._app:
            self._app._on_wake()

    def handleAppSwitch_(self, notification):
        if self._app and self._app.screen_ctx_on:
            threading.Thread(target=self._app._prefetch_context, daemon=True).start()


# ── Timing helper [P3-6] ─────────────────────────────────────────


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        elapsed = time.monotonic() - self._t0
        log.info("%s completed in %.2fs", self.label, elapsed)


# ── Inverse Text Normalization (ITN) ──────────────────────────────
from itn import _en_itn, normalize_numbers  # noqa: E402


# ── LLM Text Polish ──────────────────────────────────────────────
from text_polisher import TextPolisher, _needs_polish  # noqa: E402


# ── Utilities ─────────────────────────────────────────────────────

DEFAULT_DICT = {
    "vocabulary": [
        "Qwen",
        "MLX",
        "PyTorch",
        "Sapling",
    ],
}


def load_settings():
    """Load persisted user settings and apply config overrides."""
    global MODEL, SAMPLE_RATE, DOUBLE_CLICK_WINDOW, MAX_RECORDING_SECS
    settings = {}
    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text())
        except Exception as e:
            log.warning("Failed to parse settings.json (using defaults): %s", e, exc_info=True)
    # Validate types
    type_checks = {
        "sample_rate": int, "max_recording_secs": (int, float),
        "double_click_window": (int, float), "text_polish": bool,
        "auto_update": bool, "auto_dictionary": bool, "streaming": bool,
        "screen_context": bool, "ocr_languages": list, "model": str,
    }
    for key, expected in type_checks.items():
        if key in settings and not isinstance(settings[key], expected):
            log.warning("Invalid type for setting '%s': %s (expected %s), using default",
                        key, type(settings[key]).__name__, expected)
            settings[key] = _DEFAULTS.get(key)
    # [P5-3] Apply config overrides
    MODEL = settings.get("model", _DEFAULTS["model"])
    SAMPLE_RATE = settings.get("sample_rate", _DEFAULTS["sample_rate"])
    DOUBLE_CLICK_WINDOW = settings.get("double_click_window", _DEFAULTS["double_click_window"])
    MAX_RECORDING_SECS = settings.get("max_recording_secs", _DEFAULTS["max_recording_secs"])
    _DEFAULTS["ocr_languages"] = settings.get("ocr_languages", _DEFAULTS["ocr_languages"])
    _DEFAULTS["symbol"] = settings.get("symbol", _DEFAULTS["symbol"])
    return settings


def save_settings(settings):
    """Save user settings to disk (atomic write)."""
    try:
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(SETTINGS_PATH.parent), suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
                f.write('\n')
            os.replace(tmp_path, str(SETTINGS_PATH))
            os.chmod(str(SETTINGS_PATH), 0o600)
        except Exception:
            os.unlink(tmp_path)
            raise
    except Exception as e:
        log.warning("Failed to save settings: %s", e, exc_info=True)


def load_dictionary(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            log.warning("Bad dictionary file: %s", e, exc_info=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(DEFAULT_DICT, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp_path, str(path))
        os.chmod(str(path), 0o600)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return dict(DEFAULT_DICT)


def play_sound(name):
    path = f"/System/Library/Sounds/{name}.aiff"
    if os.path.exists(path):
        # [P2-5] start_new_session prevents zombies
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


_ICON_PATH = str(CONFIG_DIR / "icon_light.png")


def notify(title, body):
    """Send notification with VoiceInk icon."""
    try:
        icon = _ICON_PATH if os.path.exists(_ICON_PATH) else None
        rumps.notification(title, "", body, sound=False, icon=icon)
    except Exception:
        # Fallback for notifications before rumps app is running
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        safe_body = body.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(
            ["osascript", "-e", f'display notification "{safe_body}" with title "{safe_title}"'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


# ── Screen OCR ────────────────────────────────────────────────────


def capture_screens():
    """[AUDIT-19] Capture frontmost window first, then other screens.
    Returns list of CGImages — frontmost window is first (highest priority context).
    """
    if not _HAS_VISION:
        return []
    try:
        from AppKit import NSScreen, NSWorkspace

        images = []

        # 1. Capture frontmost window (most relevant context)
        frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
        pid = frontmost.processIdentifier() if frontmost else -1
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        for w in windows:
            if w.get("kCGWindowOwnerPID") == pid and w.get("kCGWindowLayer", 999) == 0:
                wid = w.get("kCGWindowNumber")
                if wid:
                    img = Quartz.CGWindowListCreateImage(
                        Quartz.CGRectNull,
                        Quartz.kCGWindowListOptionIncludingWindow,
                        wid,
                        Quartz.kCGWindowImageBoundsIgnoreFraming,
                    )
                    if img and Quartz.CGImageGetWidth(img) > 0 and Quartz.CGImageGetHeight(img) > 0:
                        images.append(img)
                break

        # 2. Capture remaining screens for additional context
        # Note: frontmost window content will be duplicated in screen captures,
        # but NER deduplicates entities by lowercase key, so the final context
        # has no duplicates. The only cost is NER processing slightly more text.
        for screen in NSScreen.screens():
            frame = screen.frame()
            rect = Quartz.CGRectMake(
                frame.origin.x, frame.origin.y,
                frame.size.width, frame.size.height,
            )
            img = Quartz.CGWindowListCreateImage(
                rect,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
            if img and Quartz.CGImageGetWidth(img) > 0 and Quartz.CGImageGetHeight(img) > 0:
                images.append(img)

        return images
    except Exception as e:
        log.warning("Screen capture failed: %s", e, exc_info=True)
        return []


def ocr_cgimage(cg_image):
    if cg_image is None or not _HAS_VISION:
        return ""
    try:
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        request.setUsesLanguageCorrection_(True)
        request.setRecognitionLanguages_(_DEFAULTS["ocr_languages"])
        success, error = handler.performRequests_error_([request], None)
        if not success:
            return ""
        lines = []
        for obs in request.results():
            if obs.confidence() < 0.5:
                continue
            candidates = obs.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())
        return "\n".join(lines)
    except Exception as e:
        log.warning("OCR failed: %s", e, exc_info=True)
        return ""


# [P3-4] OCR cache with TTL — avoid redundant OCR for rapid recordings
_ocr_cache_text = ""
_ocr_cache_time = 0.0
_OCR_CACHE_TTL = 5.0  # seconds


def get_screen_text():
    """OCR all screens in parallel and combine results. Cached for 5 seconds."""
    global _ocr_cache_text, _ocr_cache_time
    now = time.monotonic()
    if now - _ocr_cache_time < _OCR_CACHE_TTL and _ocr_cache_text:
        log.info("Screen OCR: using cached result (%d chars)", len(_ocr_cache_text))
        return _ocr_cache_text

    # [P3-3] Parallel OCR across screens
    images = capture_screens()
    if not images:
        return ""

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(images)) as pool:
        parts = list(pool.map(ocr_cgimage, images))

    result = "\n".join(p for p in parts if p)
    _ocr_cache_text = result
    _ocr_cache_time = now
    return result


from dictionary_ui import DictionaryGuard, DictionaryPopup  # noqa: E402


def _is_valid_context_term(term):
    """Check if a term is valid for ASR context. Filters OCR garbage.

    Returns True if the term passes all quality filters:
      - Length between 2 and 25 chars
      - Not too many digits (>30% ratio)
      - No embedded digits in short terms (letter-digit-letter = OCR artifact)
      - Starts with an alphabetic character (including CJK)
      - No non-alphanumeric characters except space, hyphen, apostrophe
    """
    key = term.lower()
    # Length gate
    if len(key) < 2 or len(key) > 25:
        return False
    # Reject terms with high digit ratio
    digit_count = sum(1 for c in key if c.isdigit())
    if digit_count > 0 and len(key) > 1:
        digit_ratio = digit_count / len(key)
        if digit_ratio > 0.3:
            return False
        # For short terms: reject embedded digits (letter-digit-letter = OCR artifact)
        if len(key) <= 6 and digit_count > 0:
            for j in range(1, len(key) - 1):
                if key[j].isdigit() and key[j-1].isalpha() and key[j+1].isalpha():
                    return False
    # Reject terms starting with digits, punctuation, or special chars
    # (but allow CJK characters which are not ASCII alpha)
    first = key[0]
    if not first.isalpha():
        return False
    # Reject terms with non-alphanumeric chars (OCR artifacts)
    if any(not (c.isalpha() or c.isdigit() or c in " -'") for c in key):
        return False
    return True


def _count_words(text):
    """Count words: CJK characters count as 1 word each, Latin words by spaces."""
    count = 0
    in_latin = False
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or '\uf900' <= ch <= '\ufaff':
            count += 1
            in_latin = False
        elif ch.isalpha():
            if not in_latin:
                count += 1
                in_latin = True
        else:
            in_latin = False
    return count


# ── Streaming HUD ────────────────────────────────────────────────


class StreamingHUD:
    """Floating translucent panel showing interim streaming transcription text.

    Singleton via shared(). Creates one panel per connected monitor so the
    HUD is visible regardless of which screen the user is looking at.
    All UI methods (show, update_text, dismiss) MUST be called on the main
    thread (from _periodic).
    """

    _instance = None

    def __init__(self):
        self._panels = []  # One NSPanel per screen
        self._labels = []  # One NSTextField per panel

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def show(self):
        """Create and display HUD panels on every monitor. MUST be called on main thread."""
        if self._panels:
            return
        try:
            from AppKit import (
                NSPanel, NSScreen, NSColor, NSFont, NSMakeRect,
                NSTextField, NSBackingStoreBuffered, NSVisualEffectView,
            )

            screens = NSScreen.screens()
            if not screens:
                return

            W, H = 500, 48

            for screen in screens:
                # Position at bottom-center of each screen's visible frame
                vf = screen.visibleFrame()
                x = vf.origin.x + (vf.size.width - W) / 2
                y = vf.origin.y + 80

                panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                    NSMakeRect(x, y, W, H),
                    128,  # NSWindowStyleMaskNonactivatingPanel
                    NSBackingStoreBuffered,
                    False,
                )
                panel.setLevel_(3)  # NSFloatingWindowLevel
                panel.setCollectionBehavior_(1 | 256)  # CanJoinAllSpaces | FullScreenAuxiliary
                panel.setOpaque_(False)
                panel.setBackgroundColor_(NSColor.clearColor())
                panel.setHasShadow_(True)
                panel.setAlphaValue_(0.0)
                panel.setHidesOnDeactivate_(False)
                panel.setFloatingPanel_(True)

                # HUD background
                content = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
                content.setMaterial_(13)  # HUDWindow
                content.setBlendingMode_(1)  # BehindWindow
                content.setState_(1)  # Active
                content.setWantsLayer_(True)
                content.layer().setCornerRadius_(12.0)
                content.layer().setMasksToBounds_(True)
                panel.setContentView_(content)

                # Text label
                label = NSTextField.labelWithString_("")
                label.setFrame_(NSMakeRect(16, 14, W - 32, 20))
                label.setFont_(NSFont.systemFontOfSize_(13.0))
                label.setTextColor_(NSColor.whiteColor())
                label.setLineBreakMode_(5)  # NSLineBreakByTruncatingHead
                content.addSubview_(label)
                self._labels.append(label)

                panel.orderFrontRegardless()
                panel.setAlphaValue_(0.92)
                self._panels.append(panel)

        except Exception as e:
            log.warning("StreamingHUD show failed: %s", e)
            self._panels = []
            self._labels = []

    def update_text(self, text):
        """Update displayed text on all panels. Truncates long text from the left."""
        if not self._labels:
            return
        if len(text) > 77:
            text = "..." + text[-74:]
        for label in self._labels:
            try:
                label.setStringValue_(text)
            except Exception:
                pass

    def dismiss(self):
        """Close all HUD panels. MUST be called on main thread."""
        if not self._panels:
            return
        for panel in self._panels:
            try:
                panel.close()
            except Exception:
                pass
        self._panels = []
        self._labels = []


# ── App ───────────────────────────────────────────────────────────


class VoiceInputApp(rumps.App):
    def __init__(self):
        from AppKit import NSImage

        # [BUG-8] Read symbol from _DEFAULTS which is updated by load_settings()
        symbol = _DEFAULTS["symbol"]
        icon_img = NSImage.imageWithSystemSymbolName_variableValue_accessibilityDescription_(
            symbol, 1.0, None
        )
        if icon_img:
            icon_img.setTemplate_(True)
        self._icon_image = icon_img

        # [AUDIT-13] Error icon
        error_img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "exclamationmark.triangle", None
        )
        if error_img:
            error_img.setTemplate_(True)
        self._icon_error = error_img

        super().__init__(name="VoiceInk", title=None, quit_button=None)
        self._icon_nsimage = icon_img
        self._image_view = None
        self._active_effect = None

        # [P4-5] Load persisted settings first — other init depends on it
        self._settings = load_settings()

        self.state = State.LOADING
        self.lock = threading.Lock()
        self._type_lock = threading.Lock()
        self.audio_frames = []
        self.stream = None
        self.key_down_time = 0.0
        self.dc_timer = None
        self.session = None
        self.kb = keyboard.Controller()
        self._rec_start_time = 0.0
        self._autostop_fired = False
        self._watchdog_fired = False
        self._polisher = TextPolisher()
        self._polisher.set_notify(notify)
        self._text_polish = self._settings.get("text_polish", True)
        self._auto_dictionary = self._settings.get("auto_dictionary", True)
        self._streaming = self._settings.get("streaming", True)
        self._stream_state = None
        self._stream_text = ""
        self._stream_hud_dirty = False
        self._stream_feeder_stop = None
        self._stream_feeder_thread = None
        self._dict_guard = DictionaryGuard()
        self._last_ax_inserted = None  # (text, ax_element_ref, field_value_after)
        self._correction_timer = None
        self._pending_dict_popup = None  # word to show in popup
        self._MAX_HISTORY = 30
        self._history = self._settings.get("history", [])[:self._MAX_HISTORY]

        # Usage statistics (local, privacy-preserving)
        import datetime as _dt
        today_str = _dt.date.today().isoformat()
        saved_stats = self._settings.get("stats", {})
        if saved_stats.get("today") != today_str:
            # Day rollover — reset daily counters
            saved_stats["today"] = today_str
            saved_stats["today_words"] = 0
            saved_stats["today_recordings"] = 0
        self._stats = {
            "today": saved_stats.get("today", today_str),
            "today_words": saved_stats.get("today_words", 0),
            "today_recordings": saved_stats.get("today_recordings", 0),
            "total_words": saved_stats.get("total_words", 0),
            "total_recordings": saved_stats.get("total_recordings", 0),
        }
        self._stats_dirty = True  # Ensure initial menu text is set

        # [P4-4] Configurable hotkey
        # [BUG-16] Validate hotkey name and log warning on fallback
        hotkey_name = self._settings.get("hotkey", DEFAULT_HOTKEY)
        resolved = getattr(keyboard.Key, hotkey_name, None)
        if resolved is None:
            log.warning("Invalid hotkey '%s' in settings, falling back to alt_r", hotkey_name)
            resolved = keyboard.Key.alt_r
            hotkey_name = "alt_r"
        self._hotkey = resolved
        log.info("Hotkey: %s", hotkey_name)
        self.screen_ctx_on = self._settings.get("screen_context", _HAS_VISION)
        self.screen_text = ""
        self._ocr_done = threading.Event()  # [P2-2] OCR sync
        self._ocr_done.set()
        self.dictionary = load_dictionary(DICT_PATH)

        # Microphone state — restored from settings
        self.preferred_mic_name = self._settings.get("preferred_mic", None)
        self.active_mic_id = None
        self._last_device_list = []
        self._last_device_refresh = 0.0
        if self.preferred_mic_name:
            log.info("Restored mic preference: '%s'", self.preferred_mic_name)
            # Resolve the device ID immediately
            for dev_id, name in [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]:
                if name == self.preferred_mic_name:
                    self.active_mic_id = dev_id
                    log.info("Mic resolved to device ID %d", dev_id)
                    break

        # Menu
        self.status_item = rumps.MenuItem("Loading model…")
        self.mic_menu = rumps.MenuItem("Microphone")
        self._rebuild_mic_menu()
        self.ctx_item = rumps.MenuItem(
            "Screen Context" + ("" if _HAS_VISION else " (unavailable)"),
            callback=self._toggle_ctx,
        )
        self.ctx_item.state = self.screen_ctx_on

        self.polish_item = rumps.MenuItem("Text Polish (AI)", callback=self._toggle_polish)
        self.polish_item.state = self._text_polish

        self.auto_dict_item = rumps.MenuItem("Auto-Dictionary", callback=self._toggle_auto_dict)
        self.auto_dict_item.state = self._auto_dictionary

        self.stream_item = rumps.MenuItem("Streaming Preview", callback=self._toggle_streaming)
        self.stream_item.state = self._streaming

        self.stats_menu = rumps.MenuItem("Statistics")
        self._rebuild_stats_menu()

        self.history_menu = rumps.MenuItem("Recent Transcriptions")
        self._rebuild_history_menu()

        self._auto_update = self._settings.get("auto_update", True)
        auto_update_item = rumps.MenuItem("Auto-Update", callback=self._toggle_auto_update)
        auto_update_item.state = self._auto_update

        self.hotkey_menu = rumps.MenuItem("Hotkey")
        self._rebuild_hotkey_menu()

        self.menu = [
            self.status_item,
            None,
            self.history_menu,
            self.stats_menu,
            self.mic_menu,
            self.hotkey_menu,
            self.ctx_item,
            self.polish_item,
            self.auto_dict_item,
            self.stream_item,
            rumps.MenuItem("Edit Dictionary", callback=self._edit_dict),
            None,
            rumps.MenuItem("Check for Updates", callback=self._manual_update),
            auto_update_item,
            rumps.MenuItem("Launch at Login", callback=self._toggle_login),
            rumps.MenuItem("Quit VoiceInk", callback=self._quit),
        ]

        # Set Launch at Login checkmark from plist
        self._update_login_menu_state()

        self._dock_hidden = False
        self._dict_mtime = 0.0
        self._keyboard_listener = None
        self._ner_lock = threading.Lock()
        self._update_lock = threading.Lock()
        self._last_key_event_time = 0.0
        self._last_audio_cb_time = 0.0

        # Track source file mtime for code-change detection (#10)
        self._startup_mtime = os.path.getmtime(os.path.abspath(__file__))
        self._periodic_tick = 0
        self._restart_menu_item = None  # Dynamic "Restart to Update" menu item
        self._update_downloaded = False  # Set by background auto-update thread

        # Sleep/wake observer
        self._sleeping = False
        self._wake_observer = None
        try:
            obs = _SleepWakeObserver.alloc().init()
            obs._app = self
            nc = NSWorkspace.sharedWorkspace().notificationCenter()
            nc.addObserver_selector_name_object_(
                obs, obs.handleWillSleep_, NSWorkspaceWillSleepNotification, None
            )
            nc.addObserver_selector_name_object_(
                obs, obs.handleDidWake_, NSWorkspaceDidWakeNotification, None
            )
            self._wake_observer = obs
            log.info("Sleep/wake observer registered")

            # Context pre-fetch on app switch
            try:
                nc.addObserver_selector_name_object_(
                    obs, obs.handleAppSwitch_,
                    NSWorkspaceDidActivateApplicationNotification, None
                )
                log.info("App-switch context pre-fetch registered")
            except Exception as e:
                log.warning("App-switch observer failed: %s", e, exc_info=True)
        except Exception as e:
            log.warning("Failed to register sleep/wake observer: %s", e, exc_info=True)

        threading.Thread(target=self._setup, daemon=True).start()

    def _ensure_image_view(self):
        """Set up NSImageView in status bar button for native SF Symbol effects."""
        if self._image_view is not None:
            return True
        try:
            from AppKit import NSImageView, NSMakeRect

            self._nsapp.nsstatusitem.setLength_(20)
            button = self._nsapp.nsstatusitem.button()
            button.setImage_(None)
            button.setTitle_("")

            self._image_view = NSImageView.imageViewWithImage_(self._icon_image)
            frame = button.bounds()
            self._image_view.setFrame_(
                NSMakeRect(1, 0, frame.size.width - 2, frame.size.height)
            )
            self._image_view.setImageScaling_(2)
            button.addSubview_(self._image_view)
            return True
        except AttributeError:
            return False

    # Pre-load Symbols framework ONCE
    _symbols_loaded = False
    _effect_recording = None
    _effect_processing = None
    _effect_opts_repeat = None
    _effect_opts_bounce = None

    @classmethod
    def _load_symbols(cls):
        if cls._symbols_loaded:
            return
        import objc

        objc.loadBundle(
            "Symbols",
            {},
            bundle_path="/System/Library/Frameworks/Symbols.framework",
        )
        NSSymbolVariableColorEffect = objc.lookUpClass("NSSymbolVariableColorEffect")
        NSSymbolBounceEffect = objc.lookUpClass("NSSymbolBounceEffect")
        NSSymbolEffectOptions = objc.lookUpClass("NSSymbolEffectOptions")

        cls._effect_recording = (
            NSSymbolVariableColorEffect.effect()
            .effectWithIterative()
            .effectWithDimInactiveLayers()
            .effectWithNonReversing()
        )
        bounce = NSSymbolBounceEffect.effect()
        bounce._setEffectType_(1)
        cls._effect_processing = bounce.effectWithByLayer()
        cls._effect_opts_repeat = NSSymbolEffectOptions.optionsWithRepeating()
        cls._effect_opts_bounce = NSSymbolEffectOptions.optionsWithRepeatingDelay_(0.0)
        cls._symbols_loaded = True

    def _apply_effect(self, effect_name):
        """Apply a native SF Symbol animation effect. Must be called on main thread."""
        if not self._ensure_image_view() or effect_name == self._active_effect:
            return
        self._load_symbols()
        if self._active_effect:
            self._image_view.removeAllSymbolEffects()
        if effect_name == "recording":
            self._image_view.addSymbolEffect_options_animated_(
                self._effect_recording, self._effect_opts_repeat, True
            )
        elif effect_name == "processing":
            self._image_view.addSymbolEffect_options_animated_(
                self._effect_processing, self._effect_opts_bounce, True
            )
        elif effect_name == "error":
            # [AUDIT-13] Switch to error icon
            self._image_view.setImage_(self._icon_error)
        if effect_name != "error" and self._active_effect == "error":
            # Restore normal icon when leaving error state
            self._image_view.setImage_(self._icon_image)
        self._active_effect = effect_name

    # ── Microphone management ────────────────────────────────

    def _get_input_devices(self):
        devs = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devs.append((i, d["name"]))
        return devs

    def _rebuild_mic_menu(self):
        devices = self._get_input_devices()
        self._last_device_list = devices
        try:
            self.mic_menu.clear()
        except AttributeError:
            pass
        default_item = rumps.MenuItem("System Default", callback=self._select_mic)
        default_item.state = self.preferred_mic_name is None
        self.mic_menu[default_item.title] = default_item
        for dev_id, name in devices:
            item = rumps.MenuItem(name, callback=self._select_mic)
            item.state = self.preferred_mic_name == name
            self.mic_menu[item.title] = item

    def _select_mic(self, sender):
        if sender.title == "System Default":
            self.preferred_mic_name = None
            self.active_mic_id = None
            log.info("Mic: switched to System Default")
        else:
            self.preferred_mic_name = sender.title
            for dev_id, name in self._get_input_devices():
                if name == sender.title:
                    self.active_mic_id = dev_id
                    break
            log.info("Mic: user selected '%s' (id=%s)", sender.title, self.active_mic_id)
        self._rebuild_mic_menu()
        self._save_settings()

    def _resolve_mic(self):
        if self.preferred_mic_name:
            for dev_id, name in self._get_input_devices():
                if name == self.preferred_mic_name:
                    if self.active_mic_id != dev_id:
                        self.active_mic_id = dev_id
                        log.info("Mic: preferred '%s' resolved to device ID %d", name, dev_id)
                    return
            if self.active_mic_id is not None:
                self.active_mic_id = None
                log.info(
                    "Mic: '%s' disconnected, falling back to default",
                    self.preferred_mic_name,
                )

    @rumps.timer(1)
    def _periodic(self, timer):
        """Main-thread timer: icon state, Dock hiding, mic hot-plug, dict reload, watchdog."""
        # ── Apply icon effect based on state (unified state machine) ──
        visual = self.state.visual
        self._apply_effect(visual)

        # Dictionary popup management
        popup = DictionaryPopup.shared()
        if popup._panels:
            popup.tick_main_thread()
        if self._auto_dictionary and self._pending_dict_popup and self.state == State.IDLE:
            word = self._pending_dict_popup
            self._pending_dict_popup = None
            popup.show(word, self._on_dict_popup_done)

        # Streaming HUD management
        if self._streaming:
            hud = StreamingHUD.shared()
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE) and self._stream_text:
                if not hud._panels:
                    hud.show()
                if self._stream_hud_dirty:
                    self._stream_hud_dirty = False
                    hud.update_text(self._stream_text)
            elif hud._panels:
                hud.dismiss()
                self._stream_hud_dirty = False

        # [BUG-13] Rebuild history menu on main thread
        if getattr(self, "_history_dirty", False):
            self._rebuild_history_menu()
            self._history_dirty = False

        # Update statistics submenu when dirty
        if getattr(self, "_stats_dirty", False):
            self._rebuild_stats_menu()
            self._stats_dirty = False

        # Rebuild mic menu on main thread (triggered by background device refresh)
        if getattr(self, "_mic_menu_dirty", False):
            self._rebuild_mic_menu()
            self._mic_menu_dirty = False

        # Apply pending status title updates from background threads
        pending = getattr(self, "_pending_status_title", None)
        if pending is not None:
            self.status_item.title = pending
            self._pending_status_title = None

        # [FIX-15] Update polish menu item when model fails to load
        if self._polisher._load_failed and getattr(self.polish_item, "state", False):
            self.polish_item.title = "Text Polish (AI) \u2014 unavailable"
            self.polish_item.state = False

        # Hide Dock icon (one-shot, must run on main thread)
        if not self._dock_hidden:
            from AppKit import NSApplication

            NSApplication.sharedApplication().setActivationPolicy_(1)
            self._dock_hidden = True
            log.info("Dock icon hidden")

        # Auto-stop check — NO lock on main thread (lock would deadlock with pynput)
        if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE):
            if self._rec_start_time and (time.time() - self._rec_start_time) > MAX_RECORDING_SECS:
                if not self._autostop_fired:
                    self._autostop_fired = True
                    log.warning("Max recording duration reached (%ds)", MAX_RECORDING_SECS)
                    notify("VoiceInk", f"Max {MAX_RECORDING_SECS // 60}min reached, transcribing…")
                    threading.Thread(target=self._timer_auto_stop, daemon=True).start()

        # Watchdog: detect stuck RECORDING_HOLD (keyboard listener may have died).
        # Only trigger if audio has ALSO stopped flowing — this distinguishes
        # "listener died" from "user is legitimately recording for a long time".
        if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE) and self._rec_start_time:
            hold_time = time.time() - self._rec_start_time
            audio_alive = (time.monotonic() - self._last_audio_cb_time) < 5.0
            if hold_time > 3 and not audio_alive:
                if not self._watchdog_fired:
                    self._watchdog_fired = True
                    log.warning("Watchdog: RECORDING_HOLD for %.0fs with no audio, listener likely dead", hold_time)
                    threading.Thread(target=self._watchdog_cancel, daemon=True).start()

        # Mic hot-plug: periodically reinitialize PortAudio to detect new devices
        # Run in background thread to avoid blocking main thread UI (#13)
        if (self.state == State.IDLE and self.stream is None
                and (time.monotonic() - self._last_device_refresh) > 60):
            self._last_device_refresh = time.monotonic()

            def _refresh_devices():
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception:
                    pass
                try:
                    current_devs = self._get_input_devices()
                    current_names = [name for _, name in current_devs]
                    last_names = [name for _, name in self._last_device_list]
                    if current_names != last_names:
                        self._last_device_list = current_devs
                        self._resolve_mic()
                        self._mic_menu_dirty = True
                        log.info("Audio devices changed, menu updated")
                except Exception:
                    pass

            threading.Thread(target=_refresh_devices, daemon=True).start()

        # Show restart menu item when background auto-update has downloaded
        if getattr(self, '_update_downloaded', False) and self._restart_menu_item is None:
            self._update_downloaded = False
            self._restart_notified = True
            self._restart_menu_item = rumps.MenuItem(
                "\u26a0\ufe0f Restart to Update", callback=self._restart_to_update
            )
            keys = list(self.menu.keys())
            if keys:
                self.menu.insert_before(keys[0], self._restart_menu_item)
            self._pending_status_title = "Update ready"
            notify("VoiceInk", "Update ready \u2014 click 'Restart to Update' in menu bar")
            log.info("Update downloaded, restart menu item added")

        # Detect code changes on disk (~every 10s, not every tick)
        self._periodic_tick += 1
        # Heartbeat every 300 ticks (~5 minutes at 1-second tick)
        if self._periodic_tick % 300 == 0:
            log.info("Heartbeat: state=%s, listener=%s", self.state.name, "alive" if self._keyboard_listener else "dead")
        if self._periodic_tick % 10 == 0 and not getattr(self, '_restart_notified', False):
            try:
                if os.path.getmtime(os.path.abspath(__file__)) > self._startup_mtime:
                    self._restart_notified = True
                    log.info("Code changed on disk, restart needed")
                    if self._restart_menu_item is None:
                        self._restart_menu_item = rumps.MenuItem(
                            "\u26a0\ufe0f Restart to Update", callback=self._restart_to_update
                        )
                        # Insert at top of menu
                        keys = list(self.menu.keys())
                        if keys:
                            self.menu.insert_before(keys[0], self._restart_menu_item)
                    self._pending_status_title = "Update ready"
                    notify("VoiceInk", "Code updated \u2014 click 'Restart to Update' in menu bar")
            except OSError:
                pass

        # Auto-reload dictionary
        try:
            mtime = os.path.getmtime(DICT_PATH)
        except OSError:
            return
        if mtime > self._dict_mtime:
            self._dict_mtime = mtime
            self.dictionary = load_dictionary(DICT_PATH)
            n = len(self.dictionary.get("vocabulary", []))
            log.info("Dictionary auto-reloaded (%d entries)", n)

    def _timer_auto_stop(self):
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE):
                self._stop_rec_and_transcribe()

    def _watchdog_cancel(self):
        """Force-stop a stuck recording, transcribe it, and restart the listener."""
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE):
                log.warning("Watchdog: force-stopping stuck recording, transcribing audio")
                notify("VoiceInk", "Recording auto-stopped — transcribing")
                self._stop_rec_and_transcribe()
        # Force listener restart so the auto-restart loop in _setup kicks in
        listener = self._keyboard_listener
        if listener:
            try:
                listener.stop()
            except Exception:
                pass

    # ── Setup ─────────────────────────────────────────────────

    def _setup(self):
        # Start polish model loading in parallel with ASR model
        polish_thread = None
        if self._text_polish:
            polish_thread = threading.Thread(target=self._polisher.load, daemon=True)
            polish_thread.start()

        # [P1-7] Model load failure handling
        try:
            log.info("Loading model %s", MODEL)
            from mlx_qwen3_asr import Session

            with Timer("Model loading"):
                self.session = Session(model=MODEL)
            log.info("Model loaded")
        except Exception as e:
            log.error("Model load failed: %s", e, exc_info=True)
            notify("VoiceInk", "Model failed to load — check log for details")
            # [P1-3] UI update via main thread is best-effort here
            self.status_item.title = "Model failed"
            self.state = State.ERROR  # [AUDIT-13] Show error icon
            return

        # Prevent macOS App Nap throttling
        try:
            from Foundation import NSProcessInfo
            self._activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                0x00FFFFFF,  # NSActivityUserInitiatedAllowingIdleSystemSleep
                "VoiceInk needs real-time keyboard monitoring and audio processing"
            )
            log.info("App Nap prevention enabled")
        except Exception as e:
            log.warning("App Nap prevention failed: %s", e, exc_info=True)

        self._start_ner_daemon()
        self._check_permissions()

        # [P5-5] Auto-update check (non-blocking)
        if self._auto_update:
            threading.Thread(target=self._auto_update_check, daemon=True).start()

        self.state = State.IDLE
        self.status_item.title = "Ready"
        if os.environ.pop("VOICEINK_RESTARTED", None):
            local_ver = (self._INSTALL_DIR / "VERSION").read_text().strip() if (self._INSTALL_DIR / "VERSION").exists() else "unknown"
            notify("VoiceInk", f"Updated to v{local_ver} — ready")
            log.info("Post-restart: running updated code v%s", local_ver)
        else:
            hotkey_name = self._settings.get("hotkey", DEFAULT_HOTKEY)
            hotkey_label = self._HOTKEY_LABELS.get(hotkey_name, hotkey_name)
            notify("VoiceInk", f"Ready — use {hotkey_label} key")

        # [P1-1] Keyboard listener auto-restart loop
        restart_delay = 3
        while True:
            try:
                log.info("Starting keyboard listener")
                with keyboard.Listener(
                    on_press=self._on_press, on_release=self._on_release
                ) as listener:
                    self._keyboard_listener = listener
                    start_time = time.monotonic()
                    listener.join()
                self._keyboard_listener = None
                log.warning("Keyboard listener exited")
                # Reset backoff if listener ran for at least 60s (was stable)
                if time.monotonic() - start_time > 60:
                    restart_delay = 3
            except Exception as e:
                log.error("Keyboard listener died: %s", e, exc_info=True)
            notify("VoiceInk", "Reconnecting hotkey listener...")
            self.status_item.title = "Listener error"
            self.state = State.ERROR
            time.sleep(restart_delay)
            # Exponential backoff: 3, 6, 12, 24, 48, 96, 192, max 300
            restart_delay = min(restart_delay * 2, 300)
            self.state = State.IDLE
            self.status_item.title = "Ready"

    # ── Update mechanism [P5-5] ─────────────────────────────

    _REPO = "zw-g/VoiceInk"
    _INSTALL_DIR = CONFIG_DIR

    def _check_for_update(self):
        """Check GitHub for newer version via VERSION file comparison."""
        try:
            import urllib.request

            # Read remote VERSION
            url = f"https://raw.githubusercontent.com/{self._REPO}/main/VERSION"
            with urllib.request.urlopen(url, timeout=5) as resp:
                remote_version = resp.read().decode().strip()

            # Read local VERSION
            local_version_file = self._INSTALL_DIR / "VERSION"
            local_version = local_version_file.read_text().strip() if local_version_file.exists() else "0.0.0"

            # Compare as semver tuples to handle downgrades and garbage correctly
            try:
                remote_parts = tuple(int(x) for x in remote_version.split('.'))
                local_parts = tuple(int(x) for x in local_version.split('.'))
            except (ValueError, AttributeError):
                log.warning("Invalid version format: remote=%s local=%s", remote_version, local_version)
                return False
            if remote_parts > local_parts:
                log.info("Update available: %s -> %s", local_version, remote_version)
                return True
            log.info("VoiceInk is up to date (v%s)", local_version)
            return False
        except Exception as e:
            log.warning("Update check failed: %s", e, exc_info=True)
            return False

    def _download_update(self):
        """Download and compile update without restarting. Returns True on success."""
        try:
            import urllib.request
            log.info("Downloading update...")

            has_git = (self._INSTALL_DIR / ".git").exists()
            if has_git:
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=str(self._INSTALL_DIR),
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    log.info("git pull: %s", result.stdout.strip())
                else:
                    log.warning("git pull failed, trying tarball: %s", result.stderr[:100])
                    has_git = False

            if not has_git:
                import shutil, tarfile, tempfile
                url = f"https://api.github.com/repos/{self._REPO}/tarball/main"
                tmp_dir = tempfile.mkdtemp()
                try:
                    tar_path = os.path.join(tmp_dir, "update.tar.gz")
                    urllib.request.urlretrieve(url, tar_path)
                    with tarfile.open(tar_path) as tar:
                        tar.extractall(tmp_dir, filter='data')
                    extracted = [d for d in os.listdir(tmp_dir)
                                 if os.path.isdir(os.path.join(tmp_dir, d))]
                    if extracted:
                        # Verify tarball integrity via commit SHA
                        # GitHub tarball dirs are named "owner-repo-shortsha/"
                        try:
                            sha_url = f"https://api.github.com/repos/{self._REPO}/commits/main"
                            with urllib.request.urlopen(sha_url, timeout=5) as resp:
                                expected_sha = json.loads(resp.read().decode())["sha"][:7]
                            if not extracted[0].endswith(expected_sha):
                                log.warning("Update integrity check failed: expected SHA %s, got dir %s",
                                            expected_sha, extracted[0])
                                return False
                            log.info("Update integrity verified (SHA: %s)", expected_sha)
                        except Exception as e:
                            log.warning("Could not verify update integrity: %s — aborting update", e)
                            return False
                        src_dir = os.path.join(tmp_dir, extracted[0])
                        for f in ["voice_input.py", "itn.py", "text_polisher.py",
                                  "dictionary_ui.py", "test_voice_input.py",
                                  "ner_common.swift", "ner_daemon.swift", "ner_tool.swift",
                                  "install.sh", "start.sh", "stop.sh", "uninstall.sh",
                                  "status.sh", "requirements.txt", "VERSION", "README.md",
                                  "VoiceInk.icns", "icon_light.png", "icon_dark.png", "icon_glow.png"]:
                            src = os.path.join(src_dir, f)
                            if os.path.exists(src):
                                dst = str(self._INSTALL_DIR / f)
                                tmp = dst + ".tmp"
                                shutil.copy2(src, tmp)
                                os.replace(tmp, dst)
                        # Ensure shell scripts are executable after tarball extraction
                        for sh in ["start.sh", "stop.sh", "uninstall.sh", "status.sh"]:
                            sh_path = str(self._INSTALL_DIR / sh)
                            if os.path.exists(sh_path):
                                os.chmod(sh_path, 0o755)
                    else:
                        log.warning("Tarball had no subdirectory")
                        return False
                    log.info("Updated from tarball")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            venv_pip = self._INSTALL_DIR / ".venv-py2app" / "bin" / "pip"
            req_file = self._INSTALL_DIR / "requirements.txt"
            if venv_pip.exists() and req_file.exists():
                r = subprocess.run(
                    [str(venv_pip), "install", "-q", "-r", str(req_file)],
                    capture_output=True, text=True, timeout=120,
                )
                if r.returncode != 0:
                    log.warning("pip install failed: %s", r.stderr[:200] if r.stderr else "")
                else:
                    log.info("Python dependencies updated")

            ner_common = str(self._INSTALL_DIR / "ner_common.swift")
            for src_file in ["ner_tool.swift", "ner_daemon.swift"]:
                name = src_file.replace(".swift", "")
                tmp_out = str(self._INSTALL_DIR / name) + ".new"
                r = subprocess.run(
                    ["swiftc", "-O", "-o", tmp_out,
                     ner_common, str(self._INSTALL_DIR / src_file)],
                    capture_output=True, timeout=60,
                )
                if r.returncode != 0:
                    log.error("swiftc failed for %s: %s", src_file, r.stderr[:200] if r.stderr else "")
                    try:
                        os.unlink(tmp_out)
                    except OSError:
                        pass
                    return False
                os.replace(tmp_out, str(self._INSTALL_DIR / name))
            log.info("Update downloaded and compiled successfully")
            return True
        except Exception as e:
            log.warning("Update download failed: %s", e, exc_info=True)
            return False

    def _perform_update(self):
        """Pull latest code, recompile NER tools, and restart."""
        try:
            notify("VoiceInk", "Updating... will restart shortly")
            if self._download_update():
                self._do_restart()
            else:
                notify("VoiceInk", "Update failed \u2014 check log for details")
        except Exception as e:
            log.error("Update failed: %s", e, exc_info=True)
            notify("VoiceInk", "Update failed — check log for details")

    def _manual_update(self, _):
        """Menu: Check for Updates clicked."""
        threading.Thread(target=self._do_manual_update, daemon=True).start()

    def _do_manual_update(self):
        with self._update_lock:
            if self._check_for_update():
                self._perform_update()
            else:
                notify("VoiceInk", "Already up to date")

    def _auto_update_check(self):
        """Background auto-update: download silently, show restart menu item."""
        time.sleep(10)
        with self._update_lock:
            if self._check_for_update():
                log.info("Update available — downloading silently")
                if self._download_update():
                    self._update_downloaded = True
                    log.info("Update downloaded — waiting for user to restart")
                else:
                    notify("VoiceInk", "Update available but download failed.")

    def _toggle_auto_update(self, sender):
        self._auto_update = not self._auto_update
        sender.state = self._auto_update
        self._settings["auto_update"] = self._auto_update
        self._save_settings()

    def _toggle_polish(self, sender):
        self._text_polish = not self._text_polish
        sender.state = self._text_polish
        self._settings["text_polish"] = self._text_polish
        self._save_settings()
        log.info("Text Polish: %s", "enabled" if self._text_polish else "disabled")
        if self._text_polish and not self._polisher._loaded:
            threading.Thread(target=self._polisher.load, daemon=True).start()

    def _toggle_auto_dict(self, sender):
        self._auto_dictionary = not self._auto_dictionary
        sender.state = self._auto_dictionary
        self._settings["auto_dictionary"] = self._auto_dictionary
        self._save_settings()
        log.info("Auto-Dictionary: %s", "enabled" if self._auto_dictionary else "disabled")

    def _toggle_streaming(self, sender):
        self._streaming = not self._streaming
        sender.state = self._streaming
        self._settings["streaming"] = self._streaming
        self._save_settings()
        log.info("Streaming: %s", "enabled" if self._streaming else "disabled")

    # ── Permission checks [P4-1] ─────────────────────────────

    def _check_permissions(self):
        """Check required permissions and notify user if missing."""
        import ctypes
        import ctypes.util

        # Check Accessibility permission
        security = ctypes.cdll.LoadLibrary(ctypes.util.find_library("ApplicationServices"))
        security.AXIsProcessTrusted.restype = ctypes.c_bool  # [BUG-17] Correct return type
        try:
            trusted = security.AXIsProcessTrusted()
            if not trusted:
                log.warning("Accessibility permission not granted")
                notify(
                    "VoiceInk",
                    "Grant Accessibility permission in System Settings > Privacy > Accessibility",
                )
                # Open System Settings directly
                subprocess.Popen(
                    ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                log.info("Accessibility permission: granted")
        except Exception as e:
            log.warning("Could not check Accessibility permission: %s", e, exc_info=True)

        # [BUG-3] Use CGPreflightScreenCaptureAccess (macOS 10.15.4+)
        if _HAS_VISION:
            try:
                has_access = Quartz.CGPreflightScreenCaptureAccess()
                if not has_access:
                    log.warning("Screen Recording permission not granted")
                    Quartz.CGRequestScreenCaptureAccess()
                    notify(
                        "VoiceInk",
                        "Grant Screen Recording permission for context-aware transcription",
                    )
                else:
                    log.info("Screen Recording permission: granted")
            except AttributeError:
                log.info("Screen Recording permission check not available on this macOS")

    # ── Menu callbacks ────────────────────────────────────────

    def _toggle_ctx(self, sender):
        if not _HAS_VISION:
            rumps.alert("Screen Context requires pyobjc-framework-Vision.")
            return
        self.screen_ctx_on = not self.screen_ctx_on
        sender.state = self.screen_ctx_on
        self._save_settings()

    def _update_login_menu_state(self):
        """Read plist and set the Launch at Login checkmark."""
        import plistlib

        plist_path = Path.home() / "Library/LaunchAgents/com.local.voiceinput.plist"
        try:
            with open(plist_path, "rb") as f:
                plist = plistlib.load(f)
            is_on = plist.get("RunAtLoad", False)
            self.menu["Launch at Login"].state = is_on
        except Exception:
            pass

    def _toggle_login(self, sender):
        """Toggle Launch at Login by modifying the LaunchAgent plist."""
        import plistlib

        plist_path = Path.home() / "Library/LaunchAgents/com.local.voiceinput.plist"
        if not plist_path.exists():
            rumps.alert("LaunchAgent plist not found.")
            return
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        current = plist.get("RunAtLoad", False)
        plist["RunAtLoad"] = not current
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)
        sender.state = not current
        log.info("Launch at Login: %s", "enabled" if not current else "disabled")

    # [P4-3] Transcription history — persisted, click to copy
    def _update_stats(self, word_count):
        """Increment usage statistics after a successful transcription."""
        import datetime as _dt
        today_str = _dt.date.today().isoformat()
        # Day rollover — reset daily counters
        if self._stats.get("today") != today_str:
            self._stats["today"] = today_str
            self._stats["today_words"] = 0
            self._stats["today_recordings"] = 0
        self._stats["today_words"] += word_count
        self._stats["today_recordings"] += 1
        self._stats["total_words"] += word_count
        self._stats["total_recordings"] += 1
        self._stats_dirty = True
        log.info("Stats: +%d words, today=%d/%d, total=%d/%d",
                 word_count,
                 self._stats["today_words"], self._stats["today_recordings"],
                 self._stats["total_words"], self._stats["total_recordings"])

    def _rebuild_stats_menu(self):
        """Rebuild the Statistics submenu with current counts."""
        try:
            self.stats_menu.clear()
        except AttributeError:
            pass
        tw = self._stats.get("today_words", 0)
        tr = self._stats.get("today_recordings", 0)
        aw = self._stats.get("total_words", 0)
        ar = self._stats.get("total_recordings", 0)
        today_item = rumps.MenuItem(f"Today: {tw:,} words ({tr:,} recordings)")
        total_item = rumps.MenuItem(f"All time: {aw:,} words ({ar:,} recordings)")
        self.stats_menu[today_item.title] = today_item
        self.stats_menu[total_item.title] = total_item

    def _add_to_history(self, text):
        self._history.insert(0, text)
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[: self._MAX_HISTORY]
        self._history_dirty = True  # [BUG-13] Defer menu rebuild to main thread
        self._save_settings()

    def _rebuild_history_menu(self):
        try:
            self.history_menu.clear()
        except AttributeError:
            pass
        if not self._history:
            empty = rumps.MenuItem("(empty)")
            self.history_menu[empty.title] = empty
            return
        for i, text in enumerate(self._history):
            label = text[:60] + ("…" if len(text) > 60 else "")
            # [BUG-12] Disambiguate duplicate labels
            base_label = label
            suffix = 2
            while label in self.history_menu:
                label = f"{base_label} ({suffix})"
                suffix += 1
            item = rumps.MenuItem(label, callback=self._copy_history)
            item._voiceink_full_text = text
            self.history_menu[label] = item

    def _copy_history(self, sender):
        text = getattr(sender, "_voiceink_full_text", sender.title)
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
        notify("VoiceInk", "Copied to clipboard")

    # [P4-4] Hotkey selection menu
    _HOTKEY_LABELS = {
        "alt_r": "Right Option",
        "alt_l": "Left Option",
        "cmd_r": "Right Command",
        "ctrl_r": "Right Control",
        "f18": "F18",
        "f19": "F19",
        "f20": "F20",
    }

    def _rebuild_hotkey_menu(self):
        try:
            self.hotkey_menu.clear()
        except AttributeError:
            pass
        current = self._settings.get("hotkey", DEFAULT_HOTKEY)
        for key_name, label in self._HOTKEY_LABELS.items():
            item = rumps.MenuItem(label, callback=self._select_hotkey)
            item._voiceink_key = key_name
            item.state = (key_name == current)
            self.hotkey_menu[label] = item

    def _select_hotkey(self, sender):
        key_name = getattr(sender, "_voiceink_key", DEFAULT_HOTKEY)
        self._hotkey = getattr(keyboard.Key, key_name, keyboard.Key.alt_r)
        self._settings["hotkey"] = key_name
        log.info("Hotkey changed to: %s", key_name)
        self._rebuild_hotkey_menu()
        self._save_settings()
        notify("VoiceInk", f"Hotkey changed to {sender.title}. Takes effect immediately.")

    def _save_settings(self):
        # [BUG-9] Merge into existing settings to preserve user config keys
        self._settings.update({
            "preferred_mic": self.preferred_mic_name,
            "screen_context": self.screen_ctx_on,
            "hotkey": self._settings.get("hotkey", DEFAULT_HOTKEY),
            "auto_update": self._auto_update,
            "text_polish": self._text_polish,
            "auto_dictionary": self._auto_dictionary,
            "streaming": self._streaming,
            "history": self._history[:self._MAX_HISTORY],
            "stats": self._stats,
        })
        save_settings(self._settings)

    def _edit_dict(self, _):
        subprocess.Popen(["open", str(DICT_PATH)])

    def _reload_dict(self, _):
        self.dictionary = load_dictionary(DICT_PATH)
        n = len(self.dictionary.get("vocabulary", []))
        notify("VoiceInk", f"Dictionary reloaded ({n} entries)")

    def _cleanup_resources(self):
        """Release all resources for restart or shutdown."""
        DictionaryPopup.shared().dismiss_if_active()
        StreamingHUD.shared().dismiss()
        if self._stream_feeder_stop:
            self._stream_feeder_stop.set()
        if self._stream_feeder_thread:
            self._stream_feeder_thread.join(timeout=2.0)
            self._stream_feeder_thread = None
        self._stream_state = None
        if self._correction_timer:
            self._correction_timer.cancel()
        self._cancel_timer()

        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None

        if self._wake_observer:
            try:
                NSWorkspace.sharedWorkspace().notificationCenter().removeObserver_(
                    self._wake_observer
                )
            except Exception:
                pass
            self._wake_observer = None

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if hasattr(self, "_ner_proc") and self._ner_proc:
            try:
                self._ner_proc.stdin.close()
            except Exception:
                pass
            try:
                self._ner_proc.stdout.close()
            except Exception:
                pass
            try:
                self._ner_proc.terminate()
                self._ner_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self._ner_proc.kill()
                    self._ner_proc.wait(timeout=1)
                except Exception:
                    pass
            except Exception:
                pass
            self._ner_proc = None

        if hasattr(self, "_activity") and self._activity:
            try:
                from Foundation import NSProcessInfo
                NSProcessInfo.processInfo().endActivity_(self._activity)
            except Exception:
                pass
            self._activity = None

        for handler in log.handlers:
            try:
                handler.flush()
            except Exception:
                pass

        self.state = State.IDLE

    def _restart_to_update(self, _):
        """Menu callback: restart to apply update."""
        threading.Thread(target=self._do_restart, daemon=True).start()

    def _do_restart(self):
        """Clean restart: wait for transcription, cleanup, execv."""
        # Wait for in-progress transcription (up to 30s)
        if self.state == State.PROCESSING:
            log.info("Restart: waiting for transcription to finish...")
            self._pending_status_title = "Restarting after transcription..."
            deadline = time.monotonic() + 30
            while self.state == State.PROCESSING and time.monotonic() < deadline:
                time.sleep(0.5)

        # Cancel any active recording
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE,
                              State.WAITING_DOUBLE_CLICK):
                self._cancel_timer()
                self._cancel_rec()
                self.state = State.IDLE

        log.info("Restarting VoiceInk (user-triggered)...")
        notify("VoiceInk", "Restarting...")
        os.environ["VOICEINK_RESTARTED"] = "1"

        self._cleanup_resources()
        time.sleep(0.5)

        for handler in list(log.handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass

        try:
            max_fd = os.sysconf("SC_OPEN_MAX")
            if max_fd > 65536:  # macOS returns INT64_MAX; cap to avoid OverflowError
                max_fd = 65536
        except (ValueError, OSError, OverflowError):
            max_fd = 1024
        os.closerange(3, max_fd)

        import sys as _sys
        python = _sys.executable
        script = str(self._INSTALL_DIR / "voice_input.py")
        os.execv(python, [python, script])

    # [P1-5] Graceful shutdown
    def _quit(self, _):
        log.info("Shutting down")
        self._cleanup_resources()
        rumps.quit_application()

    # ── Sleep/Wake handling ───────────────────────────────────

    def _on_will_sleep(self):
        """Pre-sleep: cancel active recording (don't transcribe — thread would be killed by sleep)."""
        log.info("System will sleep")
        self._sleeping = True
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE,
                              State.WAITING_DOUBLE_CLICK):
                self._cancel_timer()
                log.info("Sleep: cancelling recording (state=%s)", self.state.name)
                self._cancel_rec()
                self.state = State.IDLE

    def _on_wake(self):
        """Post-wake: refresh audio devices after hardware re-enumerates."""
        log.info("System did wake")

        def _wake_recovery():
            try:
                time.sleep(2)  # Wait for hardware re-enumeration
                try:
                    if self.stream is None:
                        sd._terminate()
                        sd._initialize()
                        log.info("Wake: PortAudio reinitialized")
                except Exception as e:
                    log.warning("Wake: PortAudio reinit failed: %s", e, exc_info=True)
                self._resolve_mic()
            finally:
                self._sleeping = False
                self._pending_status_title = "Ready"
                log.info("Wake: recovery complete")

        threading.Thread(target=_wake_recovery, daemon=True).start()

    # ── Recording ─────────────────────────────────────────────

    def _start_rec(self):
        DictionaryPopup.shared().dismiss_if_active()
        if self._correction_timer:
            self._correction_timer.cancel()
            self._correction_timer = None
        self._pending_dict_popup = None
        self._resolve_mic()
        self.audio_frames = []
        self._rec_start_time = time.time()
        self._autostop_fired = False
        self._watchdog_fired = False

        # [P1-2] Crash protection for mic errors
        try:
            self.stream = sd.InputStream(
                device=self.active_mic_id,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=512,  # 32ms at 16kHz — predictable VAD frame timing
                callback=self._audio_cb,
            )
            self.stream.start()
        except Exception as e:
            log.error("Mic open failed: %s", e, exc_info=True)
            notify("VoiceInk", "Microphone error — check System Settings > Privacy > Microphone")
            play_sound("Basso")
            self.state = State.IDLE
            return

        self.title = ""
        play_sound("Tink")
        log.info("Recording started")

        if self.screen_ctx_on:
            global _ocr_cache_time
            self.screen_text = ""  # [BUG-5] Clear stale context before new OCR
            _ocr_cache_time = 0.0  # Invalidate OCR cache for fresh capture
            self._ocr_done.clear()
            threading.Thread(target=self._capture_screen, daemon=True).start()

        # Start streaming preview if enabled
        if self._streaming and self.session:
            try:
                vocab = self.dictionary.get("vocabulary", [])
                stream_context = " ".join(vocab) if vocab else ""
                self._stream_state = self.session.init_streaming(
                    context=stream_context,
                    chunk_size_sec=1.0,
                )
                self._stream_text = ""
                self._stream_hud_dirty = True
                self._stream_feeder_stop = threading.Event()
                self._stream_feeder_thread = threading.Thread(
                    target=self._stream_feeder, daemon=True
                )
                self._stream_feeder_thread.start()
            except Exception as e:
                log.warning("Streaming init failed: %s", e)
                self._stream_state = None

    # [P4-6] Cancel recording without transcribing
    def _cancel_rec(self):
        if self._stream_feeder_stop:
            self._stream_feeder_stop.set()
        if self._stream_feeder_thread:
            self._stream_feeder_thread.join(timeout=2.0)
            self._stream_feeder_thread = None
        self._stream_state = None
        self._stream_text = ""
        self._stream_hud_dirty = True
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._rec_start_time = 0.0
        self.audio_frames = []
        self._ocr_done.set()  # Prevent stale OCR from affecting next recording
        self.title = ""
        play_sound("Funk")
        log.info("Recording cancelled")

    def _stop_rec_and_transcribe(self):
        # Stop streaming feeder before transcription
        if self._stream_feeder_stop:
            self._stream_feeder_stop.set()
        if self._stream_feeder_thread:
            self._stream_feeder_thread.join(timeout=2.0)
            self._stream_feeder_thread = None
        # [P1-2] Crash protection
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log.warning("Stream close error: %s", e, exc_info=True)
            self.stream = None
        self._rec_start_time = 0.0
        self.state = State.PROCESSING
        self.title = ""
        play_sound("Pop")
        log.info("Recording stopped")

        # All callers already hold self.lock — do NOT re-acquire here.
        # self.lock is a threading.Lock (non-reentrant); acquiring it again
        # from the same thread would deadlock permanently.
        frames = self.audio_frames
        self.audio_frames = []
        threading.Thread(target=self._transcribe, args=(frames,), daemon=True).start()

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio callback status: %s", status)
        self._last_audio_cb_time = time.monotonic()
        self.audio_frames.append(indata.copy())

    def _stream_feeder(self):
        """Feed audio chunks to streaming ASR. Runs on background thread."""
        last_fed = 0
        while not self._stream_feeder_stop.wait(timeout=0.5):
            state = self._stream_state
            if state is None:
                break
            with self.lock:
                frames = self.audio_frames
            n = len(frames)
            if n <= last_fed:
                continue
            new = frames[last_fed:n]
            last_fed = n
            try:
                pcm = np.concatenate(new).flatten()
                self._stream_state = self.session.feed_audio(pcm, state)
                text = self._stream_state.text
                if text != self._stream_text:
                    self._stream_text = text
                    self._stream_hud_dirty = True
            except Exception as e:
                log.debug("Stream feed error: %s", e)

    def _capture_screen(self):
        try:
            with Timer("Screen OCR"):
                self.screen_text = get_screen_text()
            log.info("Screen OCR: %d chars", len(self.screen_text))
        except Exception as e:
            self.screen_text = ""
            log.warning("Screen capture failed: %s", e, exc_info=True)
        finally:
            self._ocr_done.set()  # [P2-2] signal completion

    def _prefetch_context(self):
        """Pre-fetch screen context in background when app switches."""
        if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE, State.PROCESSING):
            return
        try:
            self.screen_text = get_screen_text()
            if self.screen_text:
                self._extract_entities(self.screen_text)  # warm NER cache
                log.debug("Context pre-fetched (%d chars)", len(self.screen_text))
        except Exception:
            pass

    # [P3-1] NER daemon — long-running process, no spawn overhead per call
    _NER_DAEMON_PATH = str(CONFIG_DIR / "ner_daemon")
    _NER_TOOL_PATH = str(CONFIG_DIR / "ner_tool")  # fallback

    def _start_ner_daemon(self):
        """Start the NER daemon subprocess. Called once during setup."""
        try:
            self._ner_proc = subprocess.Popen(
                [self._NER_DAEMON_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            # Wait for READY signal with timeout (use thread because
            # select.select doesn't work with Python's buffered text IO)
            ready_result = [None]

            def _read_ready():
                try:
                    ready_result[0] = self._ner_proc.stdout.readline().strip()
                except Exception:
                    pass

            t = threading.Thread(target=_read_ready, daemon=True)
            t.start()
            t.join(timeout=10.0)
            if t.is_alive() or ready_result[0] != "READY":
                log.warning("NER daemon startup failed (timeout or bad output: %s)", ready_result[0])
                self._ner_proc.terminate()
                self._ner_proc = None
                return False
            log.info("NER daemon started (PID %d)", self._ner_proc.pid)
            return True
        except Exception as e:
            log.warning("NER daemon start failed: %s", e, exc_info=True)
        if self._ner_proc:
            try:
                self._ner_proc.terminate()
            except Exception:
                pass
        self._ner_proc = None
        return False

    _ner_restart_count = 0
    _MAX_NER_RESTARTS = 3
    _ner_last_success = 0.0

    def _extract_entities(self, text):
        """Extract entities via NER daemon (fast) or fallback to subprocess."""
        # Reset restart counter after 5 minutes of successful operation
        if self._ner_last_success and (time.monotonic() - self._ner_last_success) > 300:
            if self._ner_restart_count > 0:
                log.info("NER restart counter reset (stable for 5+ min)")
                self._ner_restart_count = 0

        # Try to restart dead daemon (up to 3 times per stability window)
        if hasattr(self, "_ner_proc") and (self._ner_proc is None or self._ner_proc.poll() is not None):
            if self._ner_restart_count < self._MAX_NER_RESTARTS:
                log.info("NER daemon died, restarting (attempt %d)", self._ner_restart_count + 1)
                if self._start_ner_daemon():
                    self._ner_restart_count += 1
            else:
                log.warning("NER daemon exhausted restart attempts, falling back to subprocess mode")

        # Try daemon first (with lock to prevent interleaved requests)
        if hasattr(self, "_ner_proc") and self._ner_proc and self._ner_proc.poll() is None:
            with self._ner_lock:
                try:
                    with Timer("NER extraction (daemon)"):
                        self._ner_proc.stdin.write(text.replace("\n", " ") + "\n")
                        self._ner_proc.stdin.flush()
                        # Use thread for readline timeout (select doesn't work
                        # with Python's buffered text IO)
                        results = []
                        read_done = threading.Event()

                        def _read_loop():
                            try:
                                while True:
                                    line = self._ner_proc.stdout.readline().strip()
                                    if not line:
                                        break
                                    results.append(line)
                            except Exception:
                                pass
                            read_done.set()

                        reader = threading.Thread(target=_read_loop, daemon=True)
                        reader.start()
                        if not read_done.wait(timeout=5.0):
                            log.warning("NER daemon readline timed out")
                            try:
                                self._ner_proc.terminate()
                                self._ner_proc.wait(timeout=2)
                            except Exception:
                                pass
                            self._ner_proc = None
                            reader.join(timeout=2)  # Clean up the reader thread
                            return []
                        if results:
                            self._ner_last_success = time.monotonic()
                        return results
                except Exception as e:
                    log.warning("NER daemon call failed: %s", e, exc_info=True)
                    self._ner_proc = None

        # Fallback to one-shot subprocess
        try:
            with Timer("NER extraction (subprocess)"):
                result = subprocess.run(
                    [self._NER_TOOL_PATH],
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            if result.returncode == 0 and result.stdout.strip():
                return [
                    w.strip() for w in result.stdout.strip().split("\n") if w.strip()
                ]
        except Exception as e:
            log.warning("NER extraction failed: %s", e, exc_info=True)
        return []

    # Context limit: hallucination is caused by garbage terms, not quantity.
    # 274 clean terms at 53x audio ratio worked perfectly.
    # 669 garbage terms at 28x ratio caused hallucination.
    # So: filter quality aggressively, allow generous quantity.
    _MAX_CONTEXT_TERMS = 500  # generous cap; quality filtering is the real gate

    def _build_context(self, audio_duration=5.0):
        """Build context with ranked terms. Quality filtering prevents hallucination.

        Ranking priority:
          1. Dictionary terms (user-curated, always included)
          2. NER named entities from frontmost window (highest relevance)
          3. NER entities from other screens (lower relevance)

        Quality filters:
          - Reject terms > 25 chars (garbled OCR)
          - Reject terms with > 30% digits (OCR corruption)
        """
        terms = {}

        # Priority 1: Dictionary vocabulary (always included, highest rank)
        dictionary = self.dictionary
        for w in dictionary.get("vocabulary", []):
            terms[w.lower()] = w

        self._ocr_done.wait(timeout=0.5)

        if self.screen_text:
            entities = self._extract_entities(self.screen_text)
            for w in entities:
                if len(terms) >= self._MAX_CONTEXT_TERMS:
                    break
                key = w.lower()
                if not _is_valid_context_term(key):
                    continue
                # Strip trailing punctuation from OCR artifacts
                key = key.rstrip('.,;:!?。，；：！？')
                if len(key) < 2:
                    continue
                if key not in terms:
                    terms[key] = w.rstrip('.,;:!?\u2026')

        log.info("Context: %d terms (%.1fs audio)", len(terms), audio_duration)
        return " ".join(sorted(terms.values())) if terms else ""

    # ── Transcription ─────────────────────────────────────────

    def _transcribe(self, frames):
        if not frames:
            with self.lock:
                self.state = State.IDLE
            return
        audio = np.concatenate(frames).flatten()  # [BUG-6] Ensure 1D array for ASR
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            log.info("Audio too short (%.1fs), skipped", duration)
            with self.lock:
                self.state = State.IDLE
            play_sound("Funk")  # [P4-2] no-speech feedback
            return

        self._pending_status_title = "Transcribing..."

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        log.info("Audio level: RMS=%.4f peak=%.4f (%.1fs)", rms, peak, duration)

        try:
            with Timer("Transcription"):
                log.info("Transcribing %.1fs", duration)
                context = self._build_context(audio_duration=duration)
                if context:
                    log.info("Context terms: %s", context[:200])
                # [P3-5] Pass numpy array directly — no temp file needed
                result = self.session.transcribe(
                    (audio, SAMPLE_RATE), context=context
                )
            text = result.text.strip()
            if not text:
                log.info("No speech detected")
                play_sound("Funk")  # [P4-2]
                return

            # Post-processing: LLM text polish (handles numbers, fillers, symbols)
            self._pending_status_title = "Polishing..."
            if self._text_polish and _needs_polish(text):
                raw_asr = text
                with Timer("Text polish"):
                    text = self._polisher.polish(text)
                if text != raw_asr:
                    log.info("ASR raw: %s", raw_asr)
                    log.info("Polished: %s", text)
                else:
                    log.info("Result: %s", text)
            elif self._text_polish:
                log.info("Result (polish skipped): %s", text)
            else:
                log.info("Result: %s", text)
            self._add_to_history(text)
            self._update_stats(_count_words(text))
            self._type_text(text)
        except Exception as e:
            log.error("Transcription error: %s", e, exc_info=True)
            notify("VoiceInk", "Transcription error — check log for details")
            play_sound("Basso")  # [P4-2] error feedback
        finally:
            # Clear streaming state and signal HUD dismissal
            self._stream_state = None
            self._stream_text = ""
            self._stream_hud_dirty = True
            with self.lock:
                self.state = State.IDLE
            self._pending_status_title = "Ready"

    def _type_text(self, text):
        """[AUDIT-21] 3-tier hybrid: AX → CGEvent → clipboard paste."""
        try:
            with self._type_lock:
                # Tier 1: AX insertion (instant, no clipboard, ~8ms)
                if self._try_ax_insert(text):
                    log.info("Typed %d chars via AX API", len(text))
                    if self._auto_dictionary and self._last_ax_inserted:
                        self._schedule_correction_check()
                    return
                # Tier 2: CGEvent keyboard synthesis (no clipboard, universal, ~3ms/20chars)
                self._last_ax_inserted = None  # Clear stale AX ref
                if len(text) <= 200:
                    self._type_via_cgevent(text)
                    log.info("Typed %d chars via CGEvent", len(text))
                    if self._auto_dictionary:
                        time.sleep(0.05)
                        if self._try_snapshot_ax_field(text):
                            self._schedule_correction_check()
                    return
                # Tier 3: Clipboard paste (long text only)
                self._clipboard_paste(text)
                log.info("Typed %d chars via clipboard paste", len(text))
                if self._auto_dictionary:
                    time.sleep(0.1)
                    if self._try_snapshot_ax_field(text):
                        self._schedule_correction_check()
        except Exception as e:
            log.error("Text insertion failed: %s", e, exc_info=True)
            notify("VoiceInk", "Could not type text — try clicking a text field first")

    def _try_ax_insert(self, text):
        """Insert text via Accessibility API. Returns True if ACTUALLY successful.
        Some apps (Chrome web content) return success but silently ignore the write.
        We verify by checking the value changed.
        """
        try:
            from ApplicationServices import (
                AXUIElementCreateApplication,
                AXUIElementCopyAttributeValue,
                AXUIElementSetAttributeValue,
                AXUIElementIsAttributeSettable,
            )
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            app = ws.frontmostApplication()
            app_elem = AXUIElementCreateApplication(app.processIdentifier())
            err, focused = AXUIElementCopyAttributeValue(
                app_elem, "AXFocusedUIElement", None
            )
            if err != 0 or focused is None:
                return False

            err, settable = AXUIElementIsAttributeSettable(
                focused, "AXSelectedText", None
            )
            if err != 0 or not settable:
                return False

            # Read current value length before insertion
            err_before, val_before = AXUIElementCopyAttributeValue(
                focused, "AXValue", None
            )
            len_before = len(val_before) if err_before == 0 and val_before else -1

            # Attempt insertion
            err = AXUIElementSetAttributeValue(focused, "AXSelectedText", text)
            if err != 0:
                return False

            # Verify: read value after insertion — if unchanged, AX was silently ignored
            err_after, val_after = AXUIElementCopyAttributeValue(
                focused, "AXValue", None
            )
            len_after = len(val_after) if err_after == 0 and val_after else -1

            if err_before == 0 and err_after == 0 and val_before is not None and val_after is not None:
                if val_after == val_before:
                    # Value unchanged — insertion was silently ignored (e.g., Chrome web content)
                    log.debug("AX write silently ignored, falling back")
                    return False

            # Capture for correction detection
            try:
                self._last_ax_inserted = (text, focused, str(val_after) if err_after == 0 else None)
            except Exception:
                self._last_ax_inserted = None
            return True
        except Exception:
            return False

    def _try_snapshot_ax_field(self, inserted_text):
        """Read-only AX query to capture field value after CGEvent/clipboard insertion.
        Many apps support AX read even when AX write fails.
        Returns True on success, False on failure.
        """
        try:
            from ApplicationServices import (
                AXUIElementCreateApplication,
                AXUIElementCopyAttributeValue,
            )
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            app = ws.frontmostApplication()
            app_elem = AXUIElementCreateApplication(app.processIdentifier())
            err, focused = AXUIElementCopyAttributeValue(
                app_elem, "AXFocusedUIElement", None
            )
            if err != 0 or focused is None:
                log.debug("AX snapshot: no focused element")
                return False

            err, value = AXUIElementCopyAttributeValue(focused, "AXValue", None)
            if err != 0 or value is None:
                log.debug("AX snapshot: cannot read AXValue")
                return False

            self._last_ax_inserted = (inserted_text, focused, str(value))
            log.debug("AX snapshot captured (%d chars in field)", len(str(value)))
            return True
        except Exception:
            log.debug("AX snapshot failed", exc_info=True)
            return False

    def _type_via_cgevent(self, text, delay=0.003):
        """Type text character-by-character via CGEvent. No clipboard needed."""
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventKeyboardSetUnicodeString,
            CGEventPost,
            CGEventSetFlags,
            kCGSessionEventTap,
        )

        MAX_CHUNK = 20
        i = 0
        while i < len(text):
            chunk = text[i : i + MAX_CHUNK]
            i += len(chunk)
            for key_down in (True, False):
                ev = CGEventCreateKeyboardEvent(None, 0, key_down)
                CGEventKeyboardSetUnicodeString(ev, len(chunk), chunk)
                CGEventSetFlags(ev, 0)
                CGEventPost(kCGSessionEventTap, ev)
            if delay > 0:
                time.sleep(delay)

    _MAX_CLIPBOARD_BYTES = 10 * 1024 * 1024

    def _clipboard_paste(self, text):
        """Fallback: insert text via clipboard save/paste/restore."""
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()

        old_data = {}
        total_bytes = 0
        old_types = pb.types()
        if old_types:
            for t in old_types:
                d = pb.dataForType_(t)
                if d:
                    total_bytes += d.length()
                    if total_bytes > self._MAX_CLIPBOARD_BYTES:
                        log.warning("Clipboard too large (%d bytes), skipping restore", total_bytes)
                        old_data = {}
                        break
                    old_data[str(t)] = d

        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

        time.sleep(0.02)
        self.kb.press(keyboard.Key.cmd)
        self.kb.tap("v")
        self.kb.release(keyboard.Key.cmd)

        time.sleep(0.2)
        if old_data:
            pb.clearContents()
            for t, d in old_data.items():
                pb.setData_forType_(d, t)

    # ── Correction detection & dictionary learning ─────────────

    def _schedule_correction_check(self):
        """Schedule a 2.5s delayed check for user corrections after AX insertion."""
        if self._correction_timer:
            self._correction_timer.cancel()
        snapshot = self._last_ax_inserted
        if not snapshot:
            return
        self._correction_timer = threading.Timer(2.5, self._check_correction, args=(snapshot,))
        self._correction_timer.daemon = True
        self._correction_timer.start()

    def _check_correction(self, snapshot):
        """Read back the AX field and detect corrections."""
        try:
            inserted_text, ax_element, old_value = snapshot
            if not ax_element or not old_value:
                return

            from ApplicationServices import AXUIElementCopyAttributeValue
            err, current = AXUIElementCopyAttributeValue(ax_element, "AXValue", None)
            if err != 0 or current is None:
                return
            current = str(current)
            if current == old_value:
                return  # No change

            # Find our inserted text in old_value
            idx = old_value.rfind(inserted_text)
            if idx == -1:
                return
            prefix = old_value[:idx]
            suffix = old_value[idx + len(inserted_text):]

            # Extract corresponding region from current value
            if suffix:
                if current.startswith(prefix) and current.endswith(suffix):
                    corrected = current[len(prefix):-len(suffix)]
                else:
                    return
            else:
                if current.startswith(prefix):
                    corrected = current[len(prefix):]
                else:
                    return
                # Filter out continuation typing: user kept typing after insertion
                if corrected.startswith(inserted_text) and len(corrected) > len(inserted_text):
                    return

            if not corrected or corrected == inserted_text:
                return

            log.info("Correction detected: '%s' -> '%s'", inserted_text, corrected)
            self._evaluate_correction(inserted_text, corrected)
        except Exception as e:
            log.debug("Correction check failed: %s", e)
        finally:
            self._last_ax_inserted = None

    def _evaluate_correction(self, original, corrected):
        """Evaluate a correction for dictionary-worthiness."""
        # Extract the key different word
        import difflib
        orig_words = original.split()
        corr_words = corrected.split()
        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        new_words = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'insert'):
                new_words.extend(corr_words[j1:j2])
        if not new_words:
            return

        # Pick the best candidate
        candidates = [w for w in new_words if len(w) >= 2]
        special = [w for w in candidates if any(c.isupper() for c in w) or not w.isascii()]
        word = (special[0] if special else candidates[0]) if candidates else None
        if not word:
            return

        # Anti-spam check
        if not self._dict_guard.should_prompt(word, self.dictionary):
            log.debug("Dict guard rejected '%s'", word)
            return

        # LLM classification
        if not self._polisher._loaded:
            return
        vocab = self.dictionary.get("vocabulary", [])
        should_add, llm_word = self._polisher.classify_correction(original, corrected, vocab)
        if not should_add:
            log.info("LLM says NO for '%s'", word)
            return

        final_word = llm_word or word
        if not self._dict_guard.should_prompt(final_word, self.dictionary):
            return

        log.info("Queuing dictionary popup for '%s'", final_word)
        self._pending_dict_popup = final_word

    def _add_to_dictionary(self, word):
        """Add a word to dictionary.json with atomic write."""
        try:
            import tempfile as _tf
            dictionary = json.loads(DICT_PATH.read_text()) if DICT_PATH.exists() else {"vocabulary": []}
            vocab = dictionary.get("vocabulary", [])
            if word.lower() in {w.lower() for w in vocab}:
                return
            vocab.append(word)
            dictionary["vocabulary"] = vocab
            tmp_fd, tmp_path = _tf.mkstemp(dir=str(DICT_PATH.parent), suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(dictionary, f, indent=2, ensure_ascii=False)
                    f.write('\n')
                os.replace(tmp_path, str(DICT_PATH))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            log.info("Dictionary: added '%s' (%d total)", word, len(vocab))
        except Exception as e:
            log.error("Failed to add to dictionary: %s", e)

    def _on_dict_popup_done(self, word, confirmed):
        """Callback when dictionary popup finishes."""
        if confirmed:
            log.info("Dictionary: adding '%s' (countdown completed)", word)
            self._dict_guard.record_add()
            threading.Thread(target=self._add_to_dictionary, args=(word,), daemon=True).start()
            notify("VoiceInk", f"Added '{word}' to dictionary")
        else:
            log.info("Dictionary: user dismissed '%s'", word)
            self._dict_guard.record_reject(word)

    # ── Key events (state machine) ────────────────────────────

    def _cancel_timer(self):
        if self.dc_timer:
            self.dc_timer.cancel()
            self.dc_timer = None

    def _dc_timeout(self):
        with self.lock:
            if self.state == State.WAITING_DOUBLE_CLICK:
                # [P1-2] Crash protection
                try:
                    self._stop_rec_and_transcribe()
                except Exception as e:
                    log.error("Stop recording failed: %s", e, exc_info=True)
                    self.state = State.IDLE

    # [P1-2] All state transitions wrapped in try/except
    def _on_press(self, key):
        self._last_key_event_time = time.monotonic()
        # [P4-6] Escape cancels recording
        if key == keyboard.Key.esc:
            with self.lock:
                if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE, State.WAITING_DOUBLE_CLICK):
                    self._cancel_timer()
                    self._cancel_rec()
                    self.state = State.IDLE
                    self.status_item.title = "Ready"
                    log.info("Recording cancelled via Escape")
            return

        if key != self._hotkey:
            return
        with self.lock:
            try:
                if self.state == State.IDLE:
                    if self._sleeping:
                        return
                    if self.session is None:
                        play_sound("Basso")
                        notify("VoiceInk", "Model not loaded — check log for errors")
                        return
                    self.key_down_time = time.time()
                    self.state = State.RECORDING_HOLD
                    self._start_rec()
                elif self.state == State.WAITING_DOUBLE_CLICK:
                    self._cancel_timer()
                    self.state = State.RECORDING_TOGGLE
                    self.status_item.title = "Toggle recording…"
                    log.info("Toggle mode")
                elif self.state == State.RECORDING_TOGGLE:
                    self._stop_rec_and_transcribe()
                elif self.state == State.PROCESSING:
                    play_sound("Tink")
                    log.info("Hotkey pressed during processing — waiting for transcription")
            except Exception as e:
                log.error("Key press handler failed: %s", e, exc_info=True)
                self.state = State.IDLE

    def _on_release(self, key):
        self._last_key_event_time = time.monotonic()
        if key != self._hotkey:
            return
        with self.lock:
            try:
                if self.state == State.RECORDING_HOLD:
                    hold = time.time() - self.key_down_time
                    if hold < DOUBLE_CLICK_WINDOW:
                        self.state = State.WAITING_DOUBLE_CLICK
                        self.dc_timer = threading.Timer(
                            DOUBLE_CLICK_WINDOW, self._dc_timeout
                        )
                        self.dc_timer.start()
                    else:
                        self._stop_rec_and_transcribe()
            except Exception as e:
                log.error("Key release handler failed: %s", e, exc_info=True)
                self.state = State.IDLE


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    VoiceInputApp().run()
