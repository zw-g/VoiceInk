#!/usr/bin/env bash
# Stop VoiceInk cleanly (prevents KeepAlive from restarting)

LABEL="com.local.voiceinput"
PLIST="$HOME/Library/LaunchAgents/com.local.voiceinput.plist"

# Bootout completely unloads the service — launchd won't restart it
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null

echo "VoiceInk stopped."
echo "To start again: ~/.local/voice-input/start.sh"
