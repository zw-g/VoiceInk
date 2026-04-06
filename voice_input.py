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
    from AppKit import (
        NSObject,
        NSWorkspace,
        NSWorkspaceWillSleepNotification,
        NSWorkspaceDidWakeNotification,
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

    def __init__(self, threshold=0.015, silence_limit=94, min_speech=3):
        self.threshold = threshold
        self.silence_limit = silence_limit  # ~3s at 32ms/frame
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


# ── Inverse Text Normalization (ITN) ──────────────────────────────

# Chinese: fix cn2an false positives where 一 is a word, not the number 1
_CN_REVERT = re.compile(r"1(些|下子?|起|直|定|样|般|切|边|块儿?|会儿?|共|向|旦|时|味|概)")

# English number words recognized by word2number
_EN_NUM_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion", "and",
}
_EN_SCALE_WORDS = {"hundred", "thousand", "million", "billion", "trillion"}


def _en_itn(text):
    """English Inverse Text Normalization using word2number library."""
    try:
        from word2number import w2n
    except ImportError:
        return text

    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w_clean = words[i].lower().rstrip(".,;:!?").replace("-", " ").split()

        if w_clean and w_clean[0] in _EN_NUM_WORDS and w_clean[0] != "and":
            # Collect consecutive number words
            raw_span = [words[i]]
            j = i + 1
            while j < len(words):
                wj_parts = words[j].lower().rstrip(".,;:!?").replace("-", " ").split()
                if wj_parts and all(p in _EN_NUM_WORDS for p in wj_parts):
                    raw_span.append(words[j])
                    j += 1
                else:
                    break

            # Build the number phrase for word2number
            phrase = " ".join(raw_span).rstrip(".,;:!?")
            phrase_words = [w for w in phrase.lower().replace("-", " ").split() if w != "and"]

            # Only convert if: multi-word, contains scale, or followed by percent
            has_scale = any(w in _EN_SCALE_WORDS for w in phrase_words)
            followed_by_pct = j < len(words) and words[j].lower().rstrip(".,;:!?") == "percent"

            if len(phrase_words) >= 2 or has_scale or followed_by_pct:
                try:
                    value = w2n.word_to_num(phrase)
                    suffix = ""
                    if followed_by_pct:
                        suffix = "%"
                        j += 1
                    result.append(str(value) + suffix)
                    i = j
                    continue
                except ValueError:
                    pass

            result.append(words[i])
            i += 1
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def normalize_numbers(text):
    """Convert spoken number words to Arabic numerals (Chinese + English).

    Chinese: uses wetext (WeNet ITN) — context-aware, preserves idioms.
    English: uses word2number — handles multi-word numbers and percentages.
    """
    # Chinese ITN via wetext (professional, context-aware)
    try:
        from wetext import Normalizer

        if not hasattr(normalize_numbers, "_zh_itn"):
            normalize_numbers._zh_itn = Normalizer(lang="zh", operator="itn")
        text = normalize_numbers._zh_itn.normalize(text)
    except ImportError:
        # Fallback to cn2an if wetext not available
        try:
            import cn2an

            text = cn2an.transform(text, "cn2an")
            text = _CN_REVERT.sub(r"一\1", text)
        except (ImportError, Exception):
            pass
    except Exception as e:
        log.debug("wetext ITN error: %s", e)

    # English ITN
    text = _en_itn(text)
    return text


# ── LLM Text Polish ──────────────────────────────────────────────

_LLM_MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"

_LLM_SYSTEM_PROMPT = """\
You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE the original language of every word. NEVER translate between languages.
CRITICAL: Do NOT modify sentences that are already well-formed. Only fix formatting issues.

Rules:
1. Convert spoken numbers to digits:
   - Chinese: 百分之三十二→32%, 三百七十六→376, 零点五→0.5
   - English: thirty-eight→38, twelve point five→12.5
   - Dates: 二零二六年四月三号→2026年4月3号, April third→April 3rd
2. Convert math/symbols in ALL languages:
   - Chinese: 大于→>, 小于→<, 等于→=, 加→+, 减→-, 乘以→×, 除以→÷, 大于等于→≥, 小于等于→≤, 不等于→≠, 的平方→², 根号→√
   - English: is greater than→>, is less than→<, equals→=, plus→+, minus→-, times→×, divided by→÷, is greater than or equal to→≥, squared→², square root→√, to the power of→superscript
3. Remove filler words ONLY:
   Chinese: 呃, 嗯, 那个, 就是说, 然后呢
   English: um, uh, like (as filler), you know (as filler), so basically
4. Add punctuation
5. Preserve idioms (三心二意, 不管三七二十一)
6. Do NOT rephrase, reword, or modify meaningful content

Example 1: 呃就是说这个东西嗯还不错然后呢我们看看
Output 1: 这个东西还不错，我们看看

Example 2: 二零二六年四月三号下午两点半我们开会
Output 2: 2026年4月3号下午2:30我们开会

Example 3: 呃这个model的performance大概百分之九十五然后呢还不错
Output 3: 这个model的performance大概95%，还不错"""


def _needs_polish(text):
    """Quick heuristic: does this text need LLM polishing?

    Returns False for short, clean text that would come back identical
    from the LLM, saving 0.8-2.6s of latency.
    """
    # Very short text — not worth the LLM overhead
    if len(text) < 8:
        return False
    # Chinese filler words
    if re.search(r'呃|嗯|就是说|然后呢|那个', text):
        return True
    # English filler words
    if re.search(r'\bum\b|\buh\b|\bso basically\b', text, re.IGNORECASE):
        return True
    # Unconverted Chinese number words (百分之, 零点, or consecutive number characters)
    if re.search(r'百分之|零点|[一二三四五六七八九十百千万亿]{2,}', text):
        return True
    # Unconverted English number words
    if re.search(r'\b(thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b', text, re.IGNORECASE):
        return True
    # Unconverted math expressions (Chinese)
    if re.search(r'大于|小于|等于|乘以|除以|大于等于|小于等于|不等于', text):
        return True
    # Unconverted math expressions (English)
    if re.search(r'\b(greater than|less than|equals|squared|divided by)\b', text, re.IGNORECASE):
        return True
    # No obvious issues found — skip polish
    return False


class TextPolisher:
    """LLM-based text post-processor using Qwen3-8B on MLX (r4_three_targeted_examples, 95.2% benchmark)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._loaded = False
        self._load_failed = False

    def load(self):
        """Load the LLM model. Call from background thread."""
        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.sample_utils import make_sampler

            log.info("Loading text polish model %s", _LLM_MODEL_ID)
            self._model, self._tokenizer = mlx_load(_LLM_MODEL_ID)
            self._sampler = make_sampler(temp=0.3, top_p=0.8, top_k=20)
            self._loaded = True
            log.info("Text polish model loaded")
        except Exception as e:
            log.warning("Text polish model failed to load: %s", e, exc_info=True)
            self._loaded = False
            self._load_failed = True
            notify("VoiceInk", "Text polish model failed to load")

    def polish(self, text):
        """Polish text using the LLM. Returns original text on failure."""
        if not self._loaded or not text.strip():
            return text
        try:
            from mlx_lm import generate

            messages = [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text + "\n/no_think"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            raw = generate(
                self._model, self._tokenizer, prompt=prompt,
                max_tokens=max(len(text) * 3, 200),
                sampler=self._sampler, verbose=False,
            )
            # Strip thinking block if present
            clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

            # Safety: if output is empty or wildly different length, use original
            if not clean or len(clean) < len(text) * 0.3 or len(clean) > len(text) * 2:
                log.warning("Text polish output rejected (len %d→%d), using original", len(text), len(clean))
                return text
            return clean
        except Exception as e:
            log.warning("Text polish failed: %s", e, exc_info=True)
            return text


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
                    if img:
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
            if img:
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
        self._polisher = TextPolisher()
        self._text_polish = self._settings.get("text_polish", True)
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
            self.polish_item,
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
        self._last_key_event_time = 0.0
        self._last_audio_cb_time = 0.0

        # Track source file mtime for code-change detection (#10)
        self._startup_mtime = os.path.getmtime(os.path.abspath(__file__))
        self._periodic_tick = 0

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
                        log.info("Mic: preferred '%s' reconnected", name)
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

        # [BUG-13] Rebuild history menu on main thread
        if getattr(self, "_history_dirty", False):
            self._rebuild_history_menu()
            self._history_dirty = False

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
                    if current_devs != self._last_device_list:
                        self._last_device_list = current_devs
                        self._resolve_mic()
                        self._mic_menu_dirty = True
                        log.info("Audio devices changed, menu updated")
                except Exception:
                    pass

            threading.Thread(target=_refresh_devices, daemon=True).start()

        # Detect code changes on disk (~every 10s, not every tick)
        self._periodic_tick += 1
        if self._periodic_tick % 10 == 0 and not getattr(self, '_restart_notified', False):
            try:
                if os.path.getmtime(os.path.abspath(__file__)) > self._startup_mtime:
                    self._restart_notified = True
                    notify("VoiceInk", "Code updated on disk. Restart to apply changes.")
                    self._pending_status_title = "Restart needed"
                    log.info("Code changed on disk, restart needed")
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
                notify("VoiceInk", "Key release missed — transcribing anyway")
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
            notify("VoiceInk", f"Model failed: {e}")
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
        notify("VoiceInk", "Ready — use right Option key")

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
            notify("VoiceInk", f"Keyboard listener lost — restarting in {restart_delay}s")
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
                # Check if remote is actually ahead (not local ahead with unpushed commits)
                result = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", remote_sha, "HEAD"],
                    cwd=str(self._INSTALL_DIR),
                    capture_output=True,
                )
                if result.returncode == 0:
                    # remote_sha is ancestor of HEAD — local is ahead, no update needed
                    log.info("Local is ahead of remote (unpushed commits), no update needed")
                    return False
                log.info("Update available: %s -> %s", local_sha[:8], remote_sha[:8])
                return True
            log.info("VoiceInk is up to date (%s)", local_sha[:8])
            return False
        except Exception as e:
            log.warning("Update check failed: %s", e, exc_info=True)
            return False

    def _perform_update(self):
        """Pull latest code, recompile NER tools, and restart."""
        try:
            log.info("Updating VoiceInk...")
            notify("VoiceInk", "Updating... will restart shortly")

            # Capture current HEAD before pulling
            pre_pull = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self._INSTALL_DIR),
                capture_output=True,
                text=True,
            )
            local_sha = pre_pull.stdout.strip() if pre_pull.returncode == 0 else ""

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

            # Check if HEAD actually changed
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self._INSTALL_DIR),
                capture_output=True,
                text=True,
            )
            new_sha = new_head.stdout.strip() if new_head.returncode == 0 else ""
            if new_sha == local_sha:
                log.info("git pull brought no new commits, skipping restart")
                notify("VoiceInk", "Already up to date")
                return False

            # Update Python dependencies
            venv_pip = self._INSTALL_DIR / ".venv-py2app" / "bin" / "pip"
            req_file = self._INSTALL_DIR / "requirements.txt"
            if venv_pip.exists() and req_file.exists():
                r = subprocess.run(
                    [str(venv_pip), "install", "-q", "-r", str(req_file)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if r.returncode != 0:
                    log.warning("pip install failed: %s", r.stderr[:200] if r.stderr else "")
                else:
                    log.info("Python dependencies updated")

            # Recompile NER tools
            for src in ["ner_tool.swift", "ner_daemon.swift"]:
                name = src.replace(".swift", "")
                r = subprocess.run(
                    ["swiftc", "-O", "-o", str(self._INSTALL_DIR / name),
                     str(self._INSTALL_DIR / src)],
                    capture_output=True,
                    timeout=60,
                )
                if r.returncode != 0:
                    log.error("swiftc failed for %s: %s", src, r.stderr[:200] if r.stderr else "")
                    notify("VoiceInk", f"Update failed: {src} compilation error")
                    return False
            log.info("NER tools recompiled")

            # Comprehensive resource cleanup before execv
            self._cleanup_resources()

            log.info("Restarting VoiceInk...")
            notify("VoiceInk", "Updated! Restarting...")
            time.sleep(1)

            # Close all log handlers
            for handler in list(log.handlers):
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass

            # Close ALL file descriptors >= 3 to prevent leaks into new process
            try:
                max_fd = os.sysconf("SC_OPEN_MAX")
            except (ValueError, OSError):
                max_fd = 1024
            os.closerange(3, max_fd)

            import sys
            python = sys.executable
            script = str(self._INSTALL_DIR / "voice_input.py")
            os.execv(python, [python, script])

        except Exception as e:
            log.error("Update failed: %s", e, exc_info=True)
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
            "history": self._history[:self._MAX_HISTORY],
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

    # [P1-5] Graceful shutdown
    def _quit(self, _):
        log.info("Shutting down")
        self._cleanup_resources()
        rumps.quit_application()

    # ── Sleep/Wake handling ───────────────────────────────────

    def _on_will_sleep(self):
        """Pre-sleep: save active recording, close stream."""
        log.info("System will sleep")
        self._sleeping = True
        with self.lock:
            if self.state in (State.RECORDING_HOLD, State.RECORDING_TOGGLE,
                              State.WAITING_DOUBLE_CLICK):
                self._cancel_timer()
                log.info("Sleep: stopping recording (state=%s)", self.state.name)
                self._stop_rec_and_transcribe()

    def _on_wake(self):
        """Post-wake: refresh audio devices after hardware re-enumerates."""
        log.info("System did wake")

        def _wake_recovery():
            time.sleep(2)  # Wait for hardware re-enumeration
            try:
                if self.stream is None:
                    sd._terminate()
                    sd._initialize()
                    log.info("Wake: PortAudio reinitialized")
            except Exception as e:
                log.warning("Wake: PortAudio reinit failed: %s", e, exc_info=True)
            self._resolve_mic()
            self._sleeping = False
            self._pending_status_title = "Ready"
            log.info("Wake: recovery complete")

        threading.Thread(target=_wake_recovery, daemon=True).start()

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
                blocksize=512,  # 32ms at 16kHz — predictable VAD frame timing
                callback=self._audio_cb,
            )
            self.stream.start()
        except Exception as e:
            log.error("Mic open failed: %s", e, exc_info=True)
            notify("VoiceInk", f"Microphone error: {e}\nCheck System Settings > Privacy > Microphone")
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
                log.warning("Stream close error: %s", e, exc_info=True)
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
        self._last_audio_cb_time = time.monotonic()
        self.audio_frames.append(indata.copy())

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

        self._ocr_done.wait(timeout=2.0)

        if self.screen_text:
            entities = self._extract_entities(self.screen_text)
            for w in entities:
                if len(terms) >= self._MAX_CONTEXT_TERMS:
                    break
                key = w.lower()
                # Quality gate — reject garbage from OCR
                if len(key) < 2 or len(key) > 25:
                    continue
                # Reject terms with high digit ratio
                digit_count = sum(1 for c in key if c.isdigit())
                if digit_count > 0 and len(key) > 1:
                    digit_ratio = digit_count / len(key)
                    if digit_ratio > 0.3:
                        continue
                    # For short terms: reject embedded digits (letter-digit-letter = OCR artifact)
                    if len(key) <= 6 and digit_count > 0:
                        for j in range(1, len(key) - 1):
                            if key[j].isdigit() and key[j-1].isalpha() and key[j+1].isalpha():
                                digit_count = -1  # flag for rejection
                                break
                        if digit_count == -1:
                            continue
                # Reject terms starting with digits, punctuation, or special chars
                if not key[0].isalpha():
                    continue
                # Reject terms with non-alphanumeric chars (OCR artifacts)
                if any(not (c.isalpha() or c.isdigit() or c in " -'") for c in key):
                    continue
                if key not in terms:
                    terms[key] = w

        log.info("Context: %d terms (%.1fs audio)", len(terms), audio_duration)
        return " ".join(sorted(terms.values())) if terms else ""

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
            self._type_text(text)
        except Exception as e:
            log.error("Transcription error: %s", e, exc_info=True)
            notify("VoiceInk", f"Error: {e}")
            play_sound("Basso")  # [P4-2] error feedback
        finally:
            self.state = State.IDLE
            self._pending_status_title = "Ready"

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

            return True
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

        time.sleep(0.2)
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
