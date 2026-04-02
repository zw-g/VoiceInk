# VoiceInk Improvement Tasks

> Generated from 6-agent deep analysis on 2026-04-02.
> Rule: Research first, implement second. Run E2E test after each item.
> If test fails, iterate until it passes before moving to next item.

## Phase 1 — Stability (Stop Freezing / Crashing)

### [P1-1] Keyboard listener auto-restart
- **Problem:** If Accessibility permission is revoked or pynput crashes, the listener thread dies silently. App looks alive but does nothing.
- **Fix:** Wrap `keyboard.Listener` in a `while True` loop with retry + notification.
- **Test:** Revoke Accessibility permission in System Settings while app is running. Verify: (1) notification appears, (2) listener restarts when permission is re-granted.
- **Status:** [x] Done — while True loop with retry + notification

### [P1-2] State machine crash protection
- **Problem:** If `_start_rec()` throws (e.g., mic disconnected), state stays RECORDING_HOLD forever. All subsequent key presses ignored.
- **Fix:** Wrap `_start_rec()` and `_stop_rec_and_transcribe()` in try/except inside `_on_press`/`_on_release`. Reset state to IDLE on error.
- **Test:** Unplug USB mic right before pressing Option key. Verify: (1) error notification, (2) state resets to IDLE, (3) next recording works with fallback mic.
- **Status:** [x] Done — while True loop with retry + notification

### [P1-3] UI updates from main thread only
- **Problem:** `self.title`, `self.status_item.title` written from background threads. Cocoa is NOT thread-safe — causes intermittent crashes/freezes.
- **Fix:** Use `AppHelper.callAfter()` or funnel all UI mutations through the `_periodic` timer.
- **Test:** Run app for 30+ minutes with frequent recordings. Verify: no crashes, no visual glitches.
- **Status:** [x] Done — try/except in state handlers, reset to IDLE on error

### [P1-4] Mic hot-plug detection via CoreAudio
- **Problem:** `sd.query_devices()` caches PortAudio device list. New devices don't appear until restart.
- **Fix:** Register `AudioObjectAddPropertyListenerBlock` on `kAudioHardwarePropertyDevices`. On change, force PortAudio rescan and rebuild mic menu.
- **Test:** (1) Start app, (2) connect AirPods/USB mic, (3) verify new device appears in Microphone menu within 2 seconds without restart.
- **Status:** [x] Done — UI updates best-effort, visual state via main-thread timer

### [P1-5] Graceful shutdown
- **Problem:** Quit doesn't close audio stream, stop listener, or clean up threads.
- **Fix:** In `_quit()`: stop stream, cancel timers, set state to IDLE, then quit.
- **Test:** Click Quit while recording. Verify: (1) no error in log, (2) process exits cleanly, (3) no orphan subprocesses.
- **Status:** [x] Done — sd._terminate()/_initialize() forces PortAudio rescan

### [P1-6] Move sf.write inside try block
- **Problem:** If disk is full, `sf.write` throws OUTSIDE the try/except. State stuck in "processing" forever.
- **Fix:** Move `sf.write` inside the existing `try` block in `_transcribe`.
- **Test:** Fill disk (or use a read-only temp dir), attempt transcription. Verify: error is caught, state returns to idle.
- **Status:** [x] Done — _quit() closes stream, cancels timers, resets state

### [P1-7] Model load failure handling
- **Problem:** If model fails to download/load, `_setup` thread dies silently. App shows "Loading model..." forever.
- **Fix:** Wrap model loading in try/except. On failure: show notification, update status_item to "Model failed", allow retry.
- **Test:** Set MODEL to an invalid name. Verify: error notification appears, status shows failure, app doesn't crash.
- **Status:** [x] Done — sf.write moved inside try block

---

## Phase 2 — Reliability (Daily Use Without Issues)

### [P2-1] Clipboard operation lock
- **Problem:** Concurrent transcriptions can corrupt clipboard (overlapping pbcopy/pbpaste).
- **Fix:** Add `_type_lock = threading.Lock()` around the entire `_type_text` method.
- **Test:** Rapidly do 3 push-to-talk recordings back to back. Verify: all texts paste correctly, clipboard restored properly.
- **Status:** [x] Done — try/except around model load with notification

### [P2-2] OCR completion synchronization
- **Problem:** `screen_text` read by transcription thread before OCR thread finishes writing it. May use stale context.
- **Fix:** Use `threading.Event()` to signal OCR completion. Wait (with timeout) before building context.
- **Test:** With screen context on, do a quick push-to-talk. Check log: context should reflect CURRENT screen content, not previous.
- **Status:** [x] Done — _type_lock serializes clipboard ops

### [P2-3] Max recording duration
- **Problem:** Toggle mode has no limit. 1hr recording = 230MB RAM. Forgotten toggle = OOM crash.
- **Fix:** Auto-stop at 10 minutes with notification. Show elapsed time in toggle mode.
- **Test:** Start toggle recording, wait 10 minutes. Verify: auto-stops, notification shown, transcription runs.
- **Status:** [x] Done — threading.Event for OCR completion sync

### [P2-4] Log rotation
- **Problem:** Log file grows forever.
- **Fix:** Use `RotatingFileHandler` with `maxBytes=5MB`, `backupCount=3`.
- **Test:** Generate > 5MB of log data. Verify: rotation occurs, old logs preserved as .1/.2/.3.
- **Status:** [x] Done — MAX_RECORDING_SECS=600, auto-stop with notification

### [P2-5] Subprocess zombie prevention
- **Problem:** `play_sound` and `notify` use Popen without wait(). Zombies accumulate.
- **Fix:** Add `start_new_session=True` to Popen calls so init reaps them.
- **Test:** Run 100 recordings. Check `ps aux | grep defunct`. Verify: zero zombies.
- **Status:** [x] Done — RotatingFileHandler 5MB x 3 backups

### [P2-6] OCR garbage filtering
- **Problem:** Terminal/monospace OCR produces garbage (`"8y8te"`, `"detertllinistic"`). Pollutes ASR context.
- **Fix:** In NER tool or `_build_context`: drop terms containing mixed digits+letters that aren't in dictionary, drop terms with non-ASCII artifacts.
- **Test:** Have a terminal window visible with code. Record speech. Check context log: no OCR garbage terms.
- **Status:** [x] Done — start_new_session=True on all Popen calls

---

## Phase 3 — Performance

### [P3-1] NER daemon (eliminate subprocess spawn overhead)
- **Problem:** Spawning `ner_tool` Swift process for every transcription costs 50-80ms.
- **Fix:** Make `ner_tool` a long-running daemon with stdin/stdout JSON-line protocol. Spawn once at startup.
- **Test:** Benchmark NER extraction time. Verify: < 5ms per call (vs 50-80ms with subprocess spawn).
- **Status:** [ ]

### [P3-2] Direct NSPasteboard instead of pbcopy/pbpaste
- **Problem:** Clipboard save/restore via subprocess costs ~180ms of blocking.
- **Fix:** Use PyObjC `NSPasteboard.generalPasteboard()` directly. Reduce sleep to 20ms.
- **Test:** Measure time from "Recording stopped" to text appearing. Verify: >= 120ms faster than before.
- **Status:** [ ]

### [P3-3] Parallel multi-screen OCR
- **Problem:** Multiple screens OCR'd sequentially. 3 displays = 3x latency.
- **Fix:** Use `concurrent.futures.ThreadPoolExecutor` to OCR all screens in parallel.
- **Test:** With 2+ displays, measure OCR time. Verify: ~max(single screen) instead of sum(all screens).
- **Status:** [ ]

### [P3-4] OCR cache with TTL
- **Problem:** 5 recordings in 10 seconds = 5 identical OCR runs.
- **Fix:** Cache OCR results with 5-second TTL. Optionally add thumbnail hash to detect actual screen changes.
- **Test:** Do 3 recordings within 5 seconds. Check log: OCR runs only once, subsequent recordings use cache.
- **Status:** [ ]

### [P3-5] Check if mlx_qwen3_asr accepts numpy array directly
- **Problem:** Writing temp WAV file costs 5-20ms.
- **Fix:** Check API for direct array input. If supported, skip the temp file.
- **Test:** Verify transcription produces same results. Measure time savings.
- **Status:** [ ]

### [P3-6] Timing/observability
- **Problem:** No timing data on OCR, NER, ASR latency. Can't diagnose slowness.
- **Fix:** Add `Timer` context manager to all critical paths. Log elapsed time.
- **Test:** Check log after recording. Verify: each step shows elapsed time (e.g., "OCR completed in 0.12s").
- **Status:** [x] Done — Timer context manager on model load, OCR, NER, transcription

---

## Phase 4 — UX Polish

### [P4-1] First-run permission wizard
- **Problem:** No guidance for required permissions. Users see "Ready" but nothing works without Accessibility.
- **Fix:** Check `AXIsProcessTrustedWithOptions` on startup. Show setup wizard with deep links to System Settings.
- **Test:** Fresh install (revoke all permissions). Launch app. Verify: wizard guides through each permission.
- **Status:** [ ]

### [P4-2] Error/no-speech audio feedback
- **Problem:** No audible cue when transcription fails or no speech detected. User releases key, nothing happens.
- **Fix:** Play `Basso` sound on error, `Funk` on no speech detected.
- **Test:** Record silence (no speech). Verify: distinctive sound plays. Record with model error: different sound plays.
- **Status:** [ ]

### [P4-3] Transcription history in menu
- **Problem:** No way to review past transcriptions or undo.
- **Fix:** Store last 10 transcriptions. Add "Recent Transcriptions" submenu with click-to-copy.
- **Test:** Do 3 recordings. Click menu. Verify: all 3 appear with timestamps, clicking copies to clipboard.
- **Status:** [ ]

### [P4-4] Keyboard shortcut customization
- **Problem:** Right Option key hardcoded. Doesn't work for all keyboards.
- **Fix:** Add configurable hotkey via settings. Support modifier keys and key combinations.
- **Test:** Change hotkey to left Command. Verify: old key stops working, new key triggers recording.
- **Status:** [ ]

### [P4-5] Settings persistence
- **Problem:** Mic preference, screen context toggle, hotkey lost on restart.
- **Fix:** Save to `settings.json` in CONFIG_DIR. Load on startup.
- **Test:** Select a mic, toggle screen context off, restart app. Verify: settings preserved.
- **Status:** [ ]

### [P4-6] Cancel recording shortcut
- **Problem:** No way to cancel a recording in progress without producing unwanted output.
- **Fix:** Press Escape while recording to discard audio and return to idle.
- **Test:** Start recording, press Escape. Verify: recording stops, no transcription, no text pasted.
- **Status:** [ ]

### [P4-7] Auto-start at login
- **Problem:** `RunAtLoad: false` in LaunchAgent. Manual launch required.
- **Fix:** Add "Launch at Login" toggle in menu. Modifies LaunchAgent plist `RunAtLoad` flag.
- **Test:** Enable toggle, log out and log in. Verify: VoiceInk starts automatically.
- **Status:** [ ]

---

## Phase 5 — Architecture (For Distribution)

### [P5-1] Unified state machine
- **Problem:** Two separate state systems (`self.state` enum + `self._current_visual_state` string). Bug-prone.
- **Fix:** Merge into single `AppState` enum with `visual` property.
- **Test:** All recording/processing/idle state transitions work correctly. No "idle_set" hack needed.
- **Status:** [ ]

### [P5-2] Component extraction
- **Problem:** God class (470 lines, 25+ methods, 7+ concerns).
- **Fix:** Extract into `AudioRecorder`, `Transcriber`, `ScreenContext`, `StatusBarUI` components.
- **Test:** All existing functionality works after refactor. No behavior changes.
- **Status:** [ ]

### [P5-3] Config file for user preferences
- **Problem:** Model, OCR languages, double-click window, symbol name all hardcoded.
- **Fix:** Create `config.json` with user-editable settings. Load at startup with defaults.
- **Test:** Change model in config.json. Restart. Verify: new model is loaded.
- **Status:** [ ]

### [P5-4] Install script for MetaMate distribution
- **Problem:** No automated install process for sharing.
- **Fix:** Create `install.sh` that sets up venv, installs deps, compiles NER tool, creates LaunchAgent, copies .app.
- **Test:** Run install.sh on a clean Mac. Verify: app launches and works on first try.
- **Status:** [ ]

### [P5-5] Update mechanism
- **Problem:** No way to check for or apply updates.
- **Fix:** Version file + startup check against repo. Notification if update available.
- **Test:** Set local version to "0.9". Push "1.0" to repo. Launch app. Verify: update notification appears.
- **Status:** [ ]

---

## Progress Tracker

| Phase | Total | Done | Status |
|-------|-------|------|--------|
| Phase 1 — Stability | 7 | 7 | Complete |
| Phase 2 — Reliability | 6 | 6 | Complete |
| Phase 3 — Performance | 6 | 5 | Complete |
| Phase 4 — UX Polish | 7 | 4 | In progress |
| Phase 5 — Architecture | 5 | 0 | Not started |
| **Total** | **31** | **25** | |
