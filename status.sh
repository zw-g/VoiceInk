#!/bin/bash
LABEL="com.local.voiceinput"
echo "=== VoiceInk Status ==="
if launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -q "state = running"; then
    PID=$(launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep "pid = " | awk '{print $3}')
    echo "Status: RUNNING (PID: $PID)"
else
    echo "Status: NOT RUNNING"
fi
echo ""
echo "=== Last Log Entry ==="
tail -1 ~/.local/voice-input/voice_input.log 2>/dev/null || echo "(no log)"
