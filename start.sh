#!/usr/bin/env bash
# VoiceInk launcher — manages the LaunchAgent service
# This ensures only ONE instance runs with proper permissions

PLIST="$HOME/Library/LaunchAgents/com.local.voiceinput.plist"
LABEL="com.local.voiceinput"

if [[ ! -f "$PLIST" ]]; then
    echo "Error: LaunchAgent plist not found. Run install.sh first."
    exit 1
fi

# Stop any existing instance (both LaunchAgent and rogue processes)
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
pkill -f voice_input.py 2>/dev/null
sleep 1

# Load and start via LaunchAgent (proper permissions, single instance)
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null
launchctl kickstart -k "gui/$(id -u)/$LABEL" 2>/dev/null

echo "VoiceInk started via LaunchAgent."
echo "Check status: launchctl print gui/$(id -u)/$LABEL"
echo "View log: tail -f ~/.local/voice-input/voice_input.log"
