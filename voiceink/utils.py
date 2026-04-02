"""Utility functions: sounds, notifications, timer."""

import enum
import os
import subprocess
import time

from voiceink.config import log


class State(enum.Enum):
    LOADING = "loading"
    IDLE = "idle"
    RECORDING_HOLD = "recording_hold"
    WAITING_DOUBLE_CLICK = "waiting_double_click"
    RECORDING_TOGGLE = "recording_toggle"
    PROCESSING = "processing"

    @property
    def visual(self):
        if self in (State.RECORDING_HOLD, State.WAITING_DOUBLE_CLICK, State.RECORDING_TOGGLE):
            return "recording"
        if self == State.PROCESSING:
            return "processing"
        if self == State.IDLE:
            return "idle"
        return "loading"


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        elapsed = time.monotonic() - self._t0
        log.info("%s completed in %.2fs", self.label, elapsed)


def play_sound(name):
    path = f"/System/Library/Sounds/{name}.aiff"
    if os.path.exists(path):
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def notify(title, body):
    subprocess.Popen(
        ["osascript", "-e", f'display notification "{body}" with title "{title}"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
