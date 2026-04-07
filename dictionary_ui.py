"""Dictionary UI components for VoiceInk.

Contains DictionaryGuard (anti-spam logic) and DictionaryPopup (floating HUD).
"""

import logging
import threading
import time

log = logging.getLogger("voiceinput")

# Check if Vision/AppKit frameworks are available (same as voice_input.py)
try:
    from AppKit import NSObject
    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False


class DictionaryGuard:
    """Anti-spam logic for dictionary auto-additions."""

    def __init__(self):
        self._session_adds = 0
        self._rejected = set()
        self._MAX_PER_SESSION = 3
        self._MAX_DICT_SIZE = 500

    def should_prompt(self, word, dictionary):
        if not word or len(word) < 2:
            return False
        if self._session_adds >= self._MAX_PER_SESSION:
            return False
        vocab = dictionary.get("vocabulary", [])
        if len(vocab) >= self._MAX_DICT_SIZE:
            return False
        if word.lower() in {w.lower() for w in vocab}:
            return False
        if word.lower() in self._rejected:
            return False
        return True

    def record_add(self):
        self._session_adds += 1

    def record_reject(self, word):
        self._rejected.add(word.lower())


if _HAS_VISION:
    class _PopupDismissTarget(NSObject):
        """ObjC target for popup dismiss button."""
        _popup_ref = None

        def dismiss_(self, sender):
            if self._popup_ref:
                self._popup_ref._dismiss(confirmed=False)


class DictionaryPopup:
    """Floating HUD popup for dictionary add confirmation with countdown.

    Creates one panel per connected monitor so the popup is visible
    regardless of which screen the user is looking at.
    """

    _instance = None

    def __init__(self):
        self._panels = []  # One NSPanel per screen
        self._countdown = 5
        self._word = None
        self._callback = None
        self._progress_bars = []  # One per panel
        self._count_labels = []  # One per panel
        self._tick_thread = None
        self._tick_gen = 0  # Generation counter to stop stale tick threads
        self._pending_update = None  # countdown_remaining or None
        self._pending_dismiss = None  # True/False or None

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def show(self, word, on_complete):
        """Show popup on every connected monitor. MUST be called on main thread."""
        if self._panels:
            self._dismiss(confirmed=False)

        try:
            from AppKit import (
                NSPanel, NSScreen, NSColor, NSFont, NSMakeRect,
                NSTextField, NSButton,
                NSBackingStoreBuffered,
                NSVisualEffectView,
                NSProgressIndicator,
            )

            self._word = word
            self._callback = on_complete
            self._countdown = 5

            screens = NSScreen.screens()
            if not screens:
                return

            W, H = 340, 52
            display = word if len(word) <= 20 else word[:18] + ".."

            # Shared dismiss target for all panels' X buttons
            if _HAS_VISION:
                self._dismiss_target = _PopupDismissTarget.alloc().init()
                self._dismiss_target._popup_ref = self

            for screen in screens:
                # Position at bottom-center of each screen's visible frame
                vf = screen.visibleFrame()
                x = vf.origin.x + (vf.size.width - W) / 2
                y = vf.origin.y + 80

                # Borderless non-activating panel
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

                # Label
                label = NSTextField.labelWithString_(f"Add '{display}' to dictionary?")
                label.setFrame_(NSMakeRect(16, 16, 210, 20))
                label.setFont_(NSFont.systemFontOfSize_(13.0))
                label.setTextColor_(NSColor.whiteColor())
                content.addSubview_(label)

                # Progress bar (countdown indicator)
                progress = NSProgressIndicator.alloc().initWithFrame_(
                    NSMakeRect(16, 8, W - 80, 4)
                )
                progress.setStyle_(0)  # Bar
                progress.setMinValue_(0.0)
                progress.setMaxValue_(5.0)
                progress.setDoubleValue_(5.0)
                progress.setIndeterminate_(False)
                content.addSubview_(progress)
                self._progress_bars.append(progress)

                # Countdown label
                clabel = NSTextField.labelWithString_("5s")
                clabel.setFrame_(NSMakeRect(W - 58, 16, 24, 20))
                clabel.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(13.0, 0.2))
                clabel.setTextColor_(NSColor.colorWithWhite_alpha_(1.0, 0.7))
                content.addSubview_(clabel)
                self._count_labels.append(clabel)

                # Dismiss (X) button — undo / cancel
                btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 32, 14, 24, 24))
                btn.setTitle_("\u2715")
                btn.setBordered_(False)
                btn.setFont_(NSFont.systemFontOfSize_(14.0))
                if _HAS_VISION:
                    btn.setTarget_(self._dismiss_target)
                    btn.setAction_(b'dismiss:')
                content.addSubview_(btn)

                # Fade in
                panel.orderFrontRegardless()
                panel.setAlphaValue_(0.95)
                self._panels.append(panel)

            # Start countdown in background
            self._pending_update = None
            self._pending_dismiss = None
            self._tick_gen += 1
            gen = self._tick_gen

            def _tick():
                while self._countdown > 0 and self._panels and self._tick_gen == gen:
                    time.sleep(1.0)
                    self._countdown -= 1
                    if self._panels and self._tick_gen == gen:
                        self._pending_update = self._countdown
                if self._panels and self._countdown <= 0 and self._tick_gen == gen:
                    self._pending_dismiss = True

            self._tick_thread = threading.Thread(target=_tick, daemon=True)
            self._tick_thread.start()

        except Exception as e:
            log.warning("Dictionary popup failed: %s", e)
            self._panels = []
            self._progress_bars = []
            self._count_labels = []

    def tick_main_thread(self):
        """Called from _periodic to update all panels. MUST be on main thread."""
        if self._pending_update is not None:
            remaining = self._pending_update
            self._pending_update = None
            for clabel in self._count_labels:
                clabel.setStringValue_(f"{remaining}s")
            for progress in self._progress_bars:
                progress.setDoubleValue_(float(remaining))

        if self._pending_dismiss is True:
            self._pending_dismiss = None
            self._dismiss(confirmed=True)

    def _dismiss(self, confirmed):
        if not self._panels:
            return
        word = self._word
        callback = self._callback
        for panel in self._panels:
            try:
                panel.close()
            except Exception:
                pass
        self._panels = []
        self._word = None
        self._callback = None
        self._progress_bars = []
        self._count_labels = []
        self._countdown = 5
        if callback and word:
            callback(word, confirmed)

    def dismiss_if_active(self):
        """Dismiss without confirming (e.g., when recording starts)."""
        if self._panels:
            self._dismiss(confirmed=False)
