"""Configuration, settings, and dictionary management."""

import json
import logging
import logging.handlers
from pathlib import Path

# Defaults — overridable via settings.json
DEFAULTS = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "sample_rate": 16000,
    "double_click_window": 0.35,
    "max_recording_secs": 600,
    "hotkey": "alt_r",
    "ocr_languages": ["en", "zh-Hans", "zh-Hant"],
    "symbol": "waveform",
}

CONFIG_DIR = Path.home() / ".local" / "voice-input"
DICT_PATH = CONFIG_DIR / "dictionary.json"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LOG_PATH = CONFIG_DIR / "voice_input.log"

# Mutable config — set by load_settings()
MODEL = DEFAULTS["model"]
SAMPLE_RATE = DEFAULTS["sample_rate"]
DOUBLE_CLICK_WINDOW = DEFAULTS["double_click_window"]
MAX_RECORDING_SECS = DEFAULTS["max_recording_secs"]
DEFAULT_HOTKEY = DEFAULTS["hotkey"]

# Logging
_log_handler = logging.handlers.RotatingFileHandler(
    str(LOG_PATH), maxBytes=5 * 1024 * 1024, backupCount=3
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log = logging.getLogger("voiceinput")
log.addHandler(_log_handler)
log.setLevel(logging.INFO)

DEFAULT_DICT = {
    "vocabulary": ["Qwen", "MLX", "PyTorch", "Sapling"],
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
    MODEL = settings.get("model", DEFAULTS["model"])
    SAMPLE_RATE = settings.get("sample_rate", DEFAULTS["sample_rate"])
    DOUBLE_CLICK_WINDOW = settings.get("double_click_window", DEFAULTS["double_click_window"])
    MAX_RECORDING_SECS = settings.get("max_recording_secs", DEFAULTS["max_recording_secs"])
    DEFAULTS["ocr_languages"] = settings.get("ocr_languages", DEFAULTS["ocr_languages"])
    DEFAULTS["symbol"] = settings.get("symbol", DEFAULTS["symbol"])
    return settings


def save_settings(settings):
    """Save user settings to disk."""
    try:
        SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
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
