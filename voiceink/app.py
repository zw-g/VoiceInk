"""VoiceInk main application — re-exports from voice_input.py for package structure."""

# The app logic lives in voice_input.py (the monolithic entry point).
# This module enables `from voiceink.app import VoiceInputApp`.
# Full component extraction (splitting the 1100-line class) is tracked as
# a future task — the current class has tight coupling between state machine,
# UI, audio, and NER that makes clean separation non-trivial.

import sys
from pathlib import Path

# Add parent dir to path so voice_input.py can be imported
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from voice_input import VoiceInputApp  # noqa: E402, F401
