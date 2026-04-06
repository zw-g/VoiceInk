#!/usr/bin/env bash
set -euo pipefail

echo "=== VoiceInk Uninstaller ==="
echo ""

# Stop the service
echo "Stopping VoiceInk..."
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.local.voiceinput.plist 2>/dev/null || true
sleep 1

# Remove LaunchAgent
echo "Removing LaunchAgent..."
rm -f ~/Library/LaunchAgents/com.local.voiceinput.plist

# Remove app bundle
echo "Removing VoiceInk.app..."
sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true

# Ask about data
echo ""
read -p "Remove all data (settings, dictionary, logs)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing ~/.local/voice-input/..."
    rm -rf ~/.local/voice-input
    echo "Removing cached ML models..."
    rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B
    rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-MLX-4bit
else
    echo "Keeping data at ~/.local/voice-input/"
    echo "To remove later: rm -rf ~/.local/voice-input"
fi

echo ""
echo "=== VoiceInk uninstalled ==="
