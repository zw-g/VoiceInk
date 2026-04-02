#!/usr/bin/env bash
set -euo pipefail

echo "=== VoiceInk Installer ==="
echo ""

# Check requirements
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: VoiceInk requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

if ! command -v /opt/homebrew/bin/python3.13 &>/dev/null; then
    echo "Python 3.13 not found. Installing via Homebrew..."
    brew install python@3.13
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
fi

INSTALL_DIR="$HOME/.local/voice-input"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create install directory
mkdir -p "$INSTALL_DIR"

# Copy source files
echo "Copying files..."
cp "$SCRIPT_DIR/voice_input.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/ner_daemon.swift" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/ner_tool.swift" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/start.sh" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/start.sh"

# Create default dictionary if not exists
if [[ ! -f "$INSTALL_DIR/dictionary.json" ]]; then
    echo '{"vocabulary": ["Qwen", "MLX", "PyTorch"]}' > "$INSTALL_DIR/dictionary.json"
fi

# Create Python venv
echo "Setting up Python virtual environment..."
VENV="$INSTALL_DIR/.venv-py2app"
if [[ ! -d "$VENV" ]]; then
    /opt/homebrew/bin/python3.13 -m venv "$VENV"
fi

echo "Installing Python dependencies..."
"$VENV/bin/pip" install -q mlx-qwen3-asr sounddevice pynput rumps \
    pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa 2>&1 | tail -3

# Compile Swift NER tools
echo "Compiling NER tools..."
swiftc -O -o "$INSTALL_DIR/ner_tool" "$INSTALL_DIR/ner_tool.swift"
swiftc -O -o "$INSTALL_DIR/ner_daemon" "$INSTALL_DIR/ner_daemon.swift"

# Set up LaunchAgent
echo "Setting up LaunchAgent..."
PLIST="$HOME/Library/LaunchAgents/com.local.voiceinput.plist"
cat > "$PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.voiceinput</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV/bin/python</string>
        <string>$INSTALL_DIR/voice_input.py</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/voice_input.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/voice_input.log</string>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
PLISTEOF

# Copy .app to /Applications
echo "Installing VoiceInk.app..."
if [[ -d "$SCRIPT_DIR/VoiceInk.app" ]]; then
    sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true
    sudo cp -R "$SCRIPT_DIR/VoiceInk.app" /Applications/
elif [[ -d "$INSTALL_DIR/VoiceInk.app" ]]; then
    sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true
    sudo cp -R "$INSTALL_DIR/VoiceInk.app" /Applications/
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Start VoiceInk:"
echo "  Option 1: Click VoiceInk in /Applications/"
echo "  Option 2: ~/.local/voice-input/start.sh"
echo ""
echo "First launch will download the ASR model (~3.4 GB)."
echo "Grant these permissions when prompted:"
echo "  - Accessibility (System Settings > Privacy > Accessibility)"
echo "  - Microphone"
echo "  - Screen Recording"
echo ""
echo "Controls: Hold right Option to talk, release to type."
