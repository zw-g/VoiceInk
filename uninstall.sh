#!/usr/bin/env bash
set -euo pipefail

echo "=== VoiceInk Uninstaller ==="
echo ""

# Stop the app
echo "Stopping VoiceInk..."
pkill -f voice_input.py 2>/dev/null || true

# Unload LaunchAgent
PLIST="$HOME/Library/LaunchAgents/com.local.voiceinput.plist"
if [[ -f "$PLIST" ]]; then
    launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "  LaunchAgent removed"
fi

# Remove app from /Applications
if [[ -d "/Applications/VoiceInk.app" ]]; then
    sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || rm -rf ~/Applications/VoiceInk.app 2>/dev/null || true
    echo "  VoiceInk.app removed"
fi

# Remove install directory
INSTALL_DIR="$HOME/.local/voice-input"
if [[ -d "$INSTALL_DIR" ]]; then
    read -p "Remove all VoiceInk data ($INSTALL_DIR)? This includes settings and history. [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR"
        echo "  Install directory removed"
    else
        echo "  Kept $INSTALL_DIR (settings and history preserved)"
    fi
fi

echo ""
echo "=== VoiceInk Uninstalled ==="
