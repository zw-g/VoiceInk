#!/usr/bin/env bash
LABEL="com.local.voiceinput"
echo "=== VoiceInk Status ==="
if launchctl print "gui/$(id -u)/$LABEL" &>/dev/null; then
    echo "Service: registered"
    PID=$(launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | awk '/pid =/{print $3}')
    if [[ -n "$PID" && "$PID" != "0" ]]; then
        echo "Process: running (PID $PID)"
    else
        echo "Process: not running"
    fi
else
    echo "Service: not registered"
fi
echo ""
echo "=== Last Log ==="
tail -1 ~/.local/voice-input/voice_input.log 2>/dev/null || echo "(no log)"
