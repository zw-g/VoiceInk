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
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard

try:
    import Quartz
    import Vision
    from AppKit import NSWorkspace

    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False

# ── Configuration ─────────────────────────────────────────────────

# [P5-3] Defaults — overridable via settings.json
_DEFAULTS = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "sample_rate": 16000,
    "double_click_window": 0.35,
    "max_recording_secs": 600,
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


# ── State ─────────────────────────────────────────────────────────


class State(enum.Enum):
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


# ── VAD (Voice Activity Detection) [AUDIT-20] ────────────────────


class SimpleVAD:
    """RMS energy-based voice activity detection for real-time auto-stop."""

    def __init__(self, threshold=0.015, silence_limit=50, min_speech=3):
        self.threshold = threshold
        self.silence_limit = silence_limit  # ~1.6s at 32ms/frame
        self.min_speech = min_speech
        self.silence_count = 0
        self.speech_count = 0
        self.is_speaking = False

    def process_frame(self, float32_frame):
        """Process a single frame. Returns (is_speech, should_auto_stop)."""
        rms = np.sqrt(np.mean(float32_frame ** 2))

        if rms > self.threshold:
            self.speech_count += 1
            self.silence_count = 0
            if self.speech_count >= self.min_speech:
                self.is_speaking = True
        else:
            self.silence_count += 1
            self.speech_count = 0

        should_stop = self.is_speaking and self.silence_count >= self.silence_limit
        return rms > self.threshold, should_stop

    def reset(self):
        self.silence_count = 0
        self.speech_count = 0
        self.is_speaking = False


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
        except Exception:
            pass
    # [P5-3] Apply config overrides
    MODEL = settings.get("model", _DEFAULTS["model"])
    SAMPLE_RATE = settings.get("sample_rate", _DEFAULTS["sample_rate"])
    DOUBLE_CLICK_WINDOW = settings.get("double_click_window", _DEFAULTS["double_click_window"])
    MAX_RECORDING_SECS = settings.get("max_recording_secs", _DEFAULTS["max_recording_secs"])
    _DEFAULTS["ocr_languages"] = settings.get("ocr_languages", _DEFAULTS["ocr_languages"])
    _DEFAULTS["symbol"] = settings.get("symbol", _DEFAULTS["symbol"])
    return settings


def save_settings(settings):
    """Save user settings to disk."""
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Failed to save settings: %s", e)


def load_dictionary(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            log.warning("Bad dictionary file: %s", e)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_DICT, indent=2, ensure_ascii=False) + "\n")
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
        pid = frontmost.processIdentifier()
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
                    if img:
                        images.append(img)
                break

        # 2. Capture remaining screens for additional context
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
            if img:
                images.append(img)

        return images
    except Exception as e:
        log.warning("Screen capture failed: %s", e)
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
            candidates = obs.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())
        return "\n".join(lines)
    except Exception as e:
        log.warning("OCR failed: %s", e)
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
        self._vad = SimpleVAD()  # [AUDIT-20] Real-time VAD for auto-stop
        self._MAX_HISTORY = 30
        self._history = self._settings.get("history", [])[:self._MAX_HISTORY]

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
            self.mic_menu,
            self.hotkey_menu,
            self.ctx_item,
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
                        log.info("Mic: preferred '%s' reconnected", name)
                    return
            if self.active_mic_id is not None:
                self.active_mic_id = None
                log.info(
                    "Mic: '%s' disconnected, falling back to default",
                    self.preferred_mic_name,
                )

    @rumps.timer(2)
    def _periodic(self, timer):
        """Main-thread timer: icon state + Dock hiding + mic + dict."""
        # ── Apply icon effect based on state (unified state machine) ──
        visual = self.state.visual
        self._apply_effect(visual)

        # [BUG-13] Rebuild history menu on main thread
        if getattr(self, "_history_dirty", False):
            self._rebuild_history_menu()
            self._history_dirty = False

        # [BUG-2] State check INSIDE lock to prevent TOCTOU race
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE):
                if self._rec_start_time and (time.time() - self._rec_start_time) > MAX_RECORDING_SECS:
                    log.warning("Max recording duration reached (%ds)", MAX_RECORDING_SECS)
                    notify("VoiceInk", f"Max {MAX_RECORDING_SECS // 60}min reached, transcribing…")
                    self._stop_rec_and_transcribe()

        # Hide Dock icon (one-shot)
        if not self._dock_hidden:
            from AppKit import NSApplication

            NSApplication.sharedApplication().setActivationPolicy_(1)
            self._dock_hidden = True
            log.info("Dock icon hidden")

        # [P1-4] Mic hot-plug: check device count change as a lightweight signal.
        # We no longer call sd._terminate()/_initialize() which was destroying
        # PortAudio every 2 seconds and causing freezes.
        # Instead, just re-query devices (PortAudio returns cached list but
        # the count/names change when macOS notifies PortAudio of device changes).
        try:
            current_devs = self._get_input_devices()
            if current_devs != self._last_device_list:
                self._last_device_list = current_devs
                self._resolve_mic()
                self._rebuild_mic_menu()
                log.info("Audio devices changed, menu updated")
        except Exception:
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

    # ── Setup ─────────────────────────────────────────────────

    def _setup(self):
        # [P1-7] Model load failure handling
        try:
            log.info("Loading model %s", MODEL)
            from mlx_qwen3_asr import Session

            with Timer("Model loading"):
                self.session = Session(model=MODEL)
            log.info("Model loaded")
        except Exception as e:
            log.error("Model load failed: %s", e)
            notify("VoiceInk", f"Model failed: {e}")
            # [P1-3] UI update via main thread is best-effort here
            self.status_item.title = "Model failed"
            self.state = State.ERROR  # [AUDIT-13] Show error icon
            return

        self._start_ner_daemon()
        self._check_permissions()

        # [P5-5] Auto-update check (non-blocking)
        if self._auto_update:
            threading.Thread(target=self._auto_update_check, daemon=True).start()

        self.state = State.IDLE
        self.status_item.title = "Ready"
        notify("VoiceInk", "Ready — use right Option key")

        # [P1-1] Keyboard listener auto-restart loop
        while True:
            try:
                log.info("Starting keyboard listener")
                with keyboard.Listener(
                    on_press=self._on_press, on_release=self._on_release
                ) as listener:
                    listener.join()
                log.warning("Keyboard listener exited")
            except Exception as e:
                log.error("Keyboard listener died: %s", e)
            notify("VoiceInk", "Keyboard listener lost — restarting in 3s")
            self.status_item.title = "Listener error"
            self.state = State.ERROR  # [AUDIT-13]
            time.sleep(3)
            self.state = State.IDLE
            self.status_item.title = "Ready"

    # ── Update mechanism [P5-5] ─────────────────────────────

    _REPO = "zw-g/VoiceInk"
    _INSTALL_DIR = CONFIG_DIR

    def _check_for_update(self):
        """Check GitHub for new commits. Returns True if update available."""
        try:
            import urllib.request

            url = f"https://api.github.com/repos/{self._REPO}/commits/main"
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            remote_sha = data["sha"]

            # Get local SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self._INSTALL_DIR),
                capture_output=True,
                text=True,
            )
            local_sha = result.stdout.strip() if result.returncode == 0 else ""

            if remote_sha != local_sha and local_sha:
                log.info("Update available: %s -> %s", local_sha[:8], remote_sha[:8])
                return True
            log.info("VoiceInk is up to date (%s)", local_sha[:8])
            return False
        except Exception as e:
            log.warning("Update check failed: %s", e)
            return False

    def _perform_update(self):
        """Pull latest code, recompile NER tools, and restart."""
        try:
            log.info("Updating VoiceInk...")
            notify("VoiceInk", "Updating... will restart shortly")

            # Git pull
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(self._INSTALL_DIR),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                log.error("git pull failed: %s", result.stderr)
                notify("VoiceInk", f"Update failed: {result.stderr[:100]}")
                return False
            log.info("git pull: %s", result.stdout.strip())

            # Recompile NER tools
            for src in ["ner_tool.swift", "ner_daemon.swift"]:
                name = src.replace(".swift", "")
                r = subprocess.run(
                    ["swiftc", "-O", "-o", str(self._INSTALL_DIR / name),
                     str(self._INSTALL_DIR / src)],
                    capture_output=True,
                    timeout=60,
                )
                # [AUDIT-8] Check swiftc return code
                if r.returncode != 0:
                    log.error("swiftc failed for %s: %s", src, r.stderr[:200] if r.stderr else "")
                    notify("VoiceInk", f"Update warning: {src} compilation failed")
            log.info("NER tools recompiled")

            # [BUG-10] Clean up before restart
            if hasattr(self, "_ner_proc") and self._ner_proc:
                try:
                    self._ner_proc.stdin.close()
                    self._ner_proc.terminate()
                except Exception:
                    pass
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass

            log.info("Restarting VoiceInk...")
            notify("VoiceInk", "Updated! Restarting...")
            time.sleep(1)

            import sys
            python = sys.executable
            script = str(self._INSTALL_DIR / "voice_input.py")
            os.execv(python, [python, script])

        except Exception as e:
            log.error("Update failed: %s", e)
            notify("VoiceInk", f"Update failed: {e}")
            return False

    def _manual_update(self, _):
        """Menu: Check for Updates clicked."""
        threading.Thread(target=self._do_manual_update, daemon=True).start()

    def _do_manual_update(self):
        if self._check_for_update():
            self._perform_update()
        else:
            notify("VoiceInk", "Already up to date")

    def _auto_update_check(self):
        """Background auto-update: check and notify (don't restart mid-use)."""
        time.sleep(10)
        if self._check_for_update():
            # [AUDIT-9] Don't auto-restart — notify user and let them choose
            log.info("Update available — notifying user")
            notify("VoiceInk", "Update available! Click 'Check for Updates' to apply.")
            self.status_item.title = "Update available"

    def _toggle_auto_update(self, sender):
        self._auto_update = not self._auto_update
        sender.state = self._auto_update
        self._settings["auto_update"] = self._auto_update
        self._save_settings()
        log.info("Auto-update: %s", "enabled" if self._auto_update else "disabled")

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
            log.warning("Could not check Accessibility permission: %s", e)

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
            "history": self._history[:self._MAX_HISTORY],
        })
        save_settings(self._settings)

    def _edit_dict(self, _):
        subprocess.Popen(["open", str(DICT_PATH)])

    def _reload_dict(self, _):
        self.dictionary = load_dictionary(DICT_PATH)
        n = len(self.dictionary.get("vocabulary", []))
        notify("VoiceInk", f"Dictionary reloaded ({n} entries)")

    # [P1-5] Graceful shutdown
    def _quit(self, _):
        log.info("Shutting down")
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        # Kill NER daemon
        if hasattr(self, "_ner_proc") and self._ner_proc:
            try:
                self._ner_proc.stdin.close()
                self._ner_proc.terminate()
            except Exception:
                pass
        self.state = State.IDLE
        self._cancel_timer()
        rumps.quit_application()

    # ── Recording ─────────────────────────────────────────────

    def _start_rec(self):
        self._resolve_mic()
        self.audio_frames = []
        self._rec_start_time = time.time()
        self._vad.reset()

        # [P1-2] Crash protection for mic errors
        try:
            self.stream = sd.InputStream(
                device=self.active_mic_id,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_cb,
            )
            self.stream.start()
        except Exception as e:
            log.error("Mic open failed: %s", e)
            notify("VoiceInk", f"Microphone error: {e}")
            self.state = State.IDLE
            return

        self.title = ""
        play_sound("Tink")
        log.info("Recording started")

        if self.screen_ctx_on:
            self.screen_text = ""  # [BUG-5] Clear stale context before new OCR
            self._ocr_done.clear()
            threading.Thread(target=self._capture_screen, daemon=True).start()

    # [P4-6] Cancel recording without transcribing
    def _cancel_rec(self):
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
        # [P1-2] Crash protection
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log.warning("Stream close error: %s", e)
            self.stream = None
        self._rec_start_time = 0.0
        self.state = State.PROCESSING
        self.title = ""
        play_sound("Pop")
        log.info("Recording stopped")

        frames = list(self.audio_frames)
        self.audio_frames = []
        threading.Thread(target=self._transcribe, args=(frames,), daemon=True).start()

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio callback status: %s", status)
        self.audio_frames.append(indata.copy())

        # [AUDIT-20] VAD auto-stop in toggle mode
        if self.state == State.RECORDING_TOGGLE:
            _, should_stop = self._vad.process_frame(indata[:, 0])
            if should_stop:
                log.info("VAD: silence detected, auto-stopping toggle recording")
                threading.Thread(target=self._vad_auto_stop, daemon=True).start()

    def _vad_auto_stop(self):
        """Auto-stop toggle recording when VAD detects sustained silence."""
        with self.lock:
            if self.state == State.RECORDING_TOGGLE:
                self.status_item.title = "Ready"
                self._stop_rec_and_transcribe()

    def _capture_screen(self):
        try:
            with Timer("Screen OCR"):
                self.screen_text = get_screen_text()
            log.info("Screen OCR: %d chars", len(self.screen_text))
        except Exception as e:
            self.screen_text = ""
            log.warning("Screen capture failed: %s", e)
        finally:
            self._ocr_done.set()  # [P2-2] signal completion

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
            # Wait for READY signal
            ready = self._ner_proc.stdout.readline().strip()
            if ready == "READY":
                log.info("NER daemon started (PID %d)", self._ner_proc.pid)
                return True
            log.warning("NER daemon unexpected output: %s", ready)
        except Exception as e:
            log.warning("NER daemon start failed: %s", e)
        self._ner_proc = None
        return False

    _ner_restart_count = 0
    _MAX_NER_RESTARTS = 3

    def _extract_entities(self, text):
        """Extract entities via NER daemon (fast) or fallback to subprocess."""
        # [AUDIT-18] Try to restart dead daemon (up to 3 times)
        if hasattr(self, "_ner_proc") and (self._ner_proc is None or self._ner_proc.poll() is not None):
            if self._ner_restart_count < self._MAX_NER_RESTARTS:
                log.info("NER daemon died, restarting (attempt %d)", self._ner_restart_count + 1)
                if self._start_ner_daemon():
                    self._ner_restart_count += 1

        # Try daemon first
        if hasattr(self, "_ner_proc") and self._ner_proc and self._ner_proc.poll() is None:
            try:
                with Timer("NER extraction (daemon)"):
                    # Send text as single line (replace newlines with spaces)
                    self._ner_proc.stdin.write(text.replace("\n", " ") + "\n")
                    self._ner_proc.stdin.flush()
                    # Read until empty line
                    results = []
                    while True:
                        line = self._ner_proc.stdout.readline().strip()
                        if not line:
                            break
                        results.append(line)
                    return results
            except Exception as e:
                log.warning("NER daemon call failed: %s", e)
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
            log.warning("NER extraction failed: %s", e)
        return []

    _MAX_CONTEXT_TERMS = 80  # Prevent ASR hallucination from too many terms

    def _build_context(self):
        terms = {}
        dictionary = self.dictionary
        # Dictionary terms always included (highest priority)
        for w in dictionary.get("vocabulary", []):
            terms[w.lower()] = w

        self._ocr_done.wait(timeout=0.2)

        if self.screen_text:
            for w in self._extract_entities(self.screen_text):
                if len(terms) >= self._MAX_CONTEXT_TERMS:
                    break
                key = w.lower()
                # Extra garbage filter: reject obvious OCR noise
                if len(key) > 25:
                    continue  # Too long = garbled
                if sum(1 for c in key if c.isdigit()) > len(key) * 0.3:
                    continue  # Too many digits = OCR corruption
                if key not in terms:
                    terms[key] = w

        context = " ".join(sorted(terms.values())) if terms else ""
        if len(context) > 2000:
            # Hard cap: truncate to prevent model overload
            context = " ".join(context.split()[:self._MAX_CONTEXT_TERMS])
        return context

    # ── Transcription ─────────────────────────────────────────

    def _transcribe(self, frames):
        if not frames:
            self.state = State.IDLE
            return
        audio = np.concatenate(frames).flatten()  # [BUG-6] Ensure 1D array for ASR
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            log.info("Audio too short (%.1fs), skipped", duration)
            self.state = State.IDLE
            play_sound("Funk")  # [P4-2] no-speech feedback
            return

        try:
            with Timer("Transcription"):
                log.info("Transcribing %.1fs", duration)
                context = self._build_context()
                if context:
                    log.info("Context: %s", context)
                # [P3-5] Pass numpy array directly — no temp file needed
                result = self.session.transcribe(
                    (audio, SAMPLE_RATE), context=context
                )
            text = result.text.strip()
            if not text:
                log.info("No speech detected")
                play_sound("Funk")  # [P4-2]
                return

            log.info("Result: %s", text)
            self._add_to_history(text)
            self._type_text(text)
        except Exception as e:
            log.error("Transcription error: %s", e)
            notify("VoiceInk", f"Error: {e}")
            play_sound("Basso")  # [P4-2] error feedback
        finally:
            self.state = State.IDLE

    def _type_text(self, text):
        """[AUDIT-21] 3-tier hybrid: AX → CGEvent → clipboard paste."""
        with self._type_lock:
            # Tier 1: AX insertion (instant, no clipboard, ~8ms)
            if self._try_ax_insert(text):
                log.info("Typed %d chars via AX API", len(text))
                return
            # Tier 2: CGEvent keyboard synthesis (no clipboard, universal, ~3ms/20chars)
            if len(text) <= 200:
                self._type_via_cgevent(text)
                log.info("Typed %d chars via CGEvent", len(text))
                return
            # Tier 3: Clipboard paste (long text only)
            self._clipboard_paste(text)
            log.info("Typed %d chars via clipboard paste", len(text))

    def _try_ax_insert(self, text):
        """Insert text via Accessibility API. Returns True if successful."""
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
            err = AXUIElementSetAttributeValue(focused, "AXSelectedText", text)
            return err == 0
        except Exception:
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
            # Don't split surrogate pairs (emoji, rare CJK)
            if chunk and ord(chunk[-1]) >= 0xD800 and ord(chunk[-1]) <= 0xDBFF:
                chunk = text[i : i + MAX_CHUNK - 1]
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

        time.sleep(0.3)
        if old_data:
            pb.clearContents()
            for t, d in old_data.items():
                pb.setData_forType_(d, t)

    # ── Key events (state machine) ────────────────────────────

    def _cancel_timer(self):
        if self.dc_timer:
            self.dc_timer.cancel()
            self.dc_timer = None

    def _dc_timeout(self):
        with self.lock:
            if self.state == State.WAITING_DOUBLE_CLICK:
                self.state = State.IDLE
                # [P1-2] Crash protection
                try:
                    self._stop_rec_and_transcribe()
                except Exception as e:
                    log.error("Stop recording failed: %s", e)
                    self.state = State.IDLE

    # [P1-2] All state transitions wrapped in try/except
    def _on_press(self, key):
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
                    if self.session is None:
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
                    self.state = State.IDLE
                    self.status_item.title = "Ready"
                    self._stop_rec_and_transcribe()
            except Exception as e:
                log.error("Key press handler failed: %s", e)
                self.state = State.IDLE

    def _on_release(self, key):
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
                        self.state = State.IDLE
                        self._stop_rec_and_transcribe()
            except Exception as e:
                log.error("Key release handler failed: %s", e)
                self.state = State.IDLE


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    VoiceInputApp().run()
