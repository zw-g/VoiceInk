#!/usr/bin/env bash
set -euo pipefail

echo "=== VoiceInk Installer ==="
echo ""

# [AUDIT-7] Check all prerequisites with clear error messages
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: VoiceInk requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

# Check macOS version (requires 14+)
MACOS_VER=$(sw_vers -productVersion | cut -d. -f1)
if [[ "$MACOS_VER" -lt 14 ]]; then
    echo "Error: VoiceInk requires macOS 14 (Sonoma) or newer. You have $(sw_vers -productVersion)."
    exit 1
fi

if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew is required. Install from https://brew.sh"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

if ! command -v swiftc &>/dev/null; then
    echo "Error: Xcode Command Line Tools required for Swift compilation."
    echo "  Run: xcode-select --install"
    exit 1
fi

PYTHON313="$(brew --prefix python@3.13 2>/dev/null)/bin/python3.13"
if [[ ! -x "$PYTHON313" ]]; then
    echo "Python 3.13 not found. Installing via Homebrew..."
    brew install python@3.13
    PYTHON313="$(brew --prefix python@3.13)/bin/python3.13"
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
for f in voice_input.py itn.py text_polisher.py dictionary_ui.py test_voice_input.py ner_common.swift ner_daemon.swift ner_tool.swift start.sh stop.sh uninstall.sh status.sh requirements.txt VERSION VoiceInk.icns icon_light.png icon_dark.png icon_glow.png; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        cp "$SCRIPT_DIR/$f" "$INSTALL_DIR/"
    fi
done
chmod +x "$INSTALL_DIR/start.sh" "$INSTALL_DIR/stop.sh" "$INSTALL_DIR/uninstall.sh" "$INSTALL_DIR/status.sh"

# Create default dictionary if not exists
if [[ ! -f "$INSTALL_DIR/dictionary.json" ]]; then
    echo '{"vocabulary": ["Qwen", "MLX", "PyTorch", "Sapling"]}' > "$INSTALL_DIR/dictionary.json"
fi

# Create Python venv
echo "Setting up Python virtual environment..."
VENV="$INSTALL_DIR/.venv-py2app"
if [[ ! -d "$VENV" ]]; then
    "$PYTHON313" -m venv "$VENV"
fi

# [AUDIT-5] Install pinned dependencies from requirements.txt
echo "Installing Python dependencies..."
if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
    "$VENV/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt" 2>&1 | tail -5
else
    "$VENV/bin/pip" install -q mlx-qwen3-asr sounddevice pynput rumps \
        pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa 2>&1 | tail -5
fi

# Pre-download ML models (cached in ~/.cache/huggingface/hub/)
echo "Downloading ML models (this may take a few minutes on first install)..."
echo "  Downloading ASR model (Qwen3-ASR-1.7B, ~3.4 GB)..."
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B')" 2>&1 | tail -2
echo "  Downloading text polish model (Qwen3-8B-MLX-4bit, ~4.1 GB)..."
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-MLX-4bit')" 2>&1 | tail -2
echo "Models ready."

# [AUDIT-8] Compile Swift NER tools with error checking
echo "Compiling NER tools..."
if ! swiftc -O -o "$INSTALL_DIR/ner_tool" "$INSTALL_DIR/ner_common.swift" "$INSTALL_DIR/ner_tool.swift"; then
    echo "Error: Failed to compile ner_tool.swift"
    exit 1
fi
if ! swiftc -O -o "$INSTALL_DIR/ner_daemon" "$INSTALL_DIR/ner_common.swift" "$INSTALL_DIR/ner_daemon.swift"; then
    echo "Error: Failed to compile ner_daemon.swift"
    exit 1
fi

# [AUDIT-4+6] Set up LaunchAgent with PATH, KeepAlive, WorkingDirectory
echo "Setting up LaunchAgent..."
PLIST="$HOME/Library/LaunchAgents/com.local.voiceinput.plist"

# Preserve user's RunAtLoad preference if plist already exists
RUN_AT_LOAD="false"
if [[ -f "$PLIST" ]]; then
    if /usr/libexec/PlistBuddy -c "Print :RunAtLoad" "$PLIST" 2>/dev/null | grep -q "true"; then
        RUN_AT_LOAD="true"
    fi
fi

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
    <$RUN_AT_LOAD/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
PLISTEOF

# Build lightweight .app wrapper (no py2app needed)
echo "Creating VoiceInk.app..."
APP_DIR="$INSTALL_DIR/VoiceInk.app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Launcher script
cat > "$APP_DIR/Contents/MacOS/VoiceInk" << 'LAUNCHEREOF'
#!/bin/bash
INSTALL_DIR="$HOME/.local/voice-input"
VENV="$INSTALL_DIR/.venv-py2app"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "$VENV/bin/python" "$INSTALL_DIR/voice_input.py"
LAUNCHEREOF
chmod +x "$APP_DIR/Contents/MacOS/VoiceInk"

# Info.plist
cat > "$APP_DIR/Contents/Info.plist" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>VoiceInk</string>
    <key>CFBundleDisplayName</key>
    <string>VoiceInk</string>
    <key>CFBundleIdentifier</key>
    <string>com.local.voiceink</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleExecutable</key>
    <string>VoiceInk</string>
    <key>CFBundleIconFile</key>
    <string>VoiceInk</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>VoiceInk needs microphone access for voice-to-text transcription.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>VoiceInk needs accessibility access for keyboard shortcuts and text insertion.</string>
</dict>
</plist>
PLISTEOF

# Copy icon
if [[ -f "$INSTALL_DIR/VoiceInk.icns" ]]; then
    cp "$INSTALL_DIR/VoiceInk.icns" "$APP_DIR/Contents/Resources/VoiceInk.icns"
fi

# Install to /Applications
sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true
sudo cp -R "$APP_DIR" /Applications/ 2>/dev/null || echo "  Could not copy to /Applications (no sudo). You can drag VoiceInk.app from $APP_DIR manually."
echo "VoiceInk.app installed to /Applications/"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Start VoiceInk:"
echo "  Option 1: Open VoiceInk from /Applications/ (recommended)"
echo "  Option 2: ~/.local/voice-input/start.sh"
echo ""
echo "To start automatically on login:"
echo "  launchctl load ~/Library/LaunchAgents/com.local.voiceinput.plist"
echo ""
echo "IMPORTANT: On first launch, macOS will ask for permissions."
echo "  Grant ALL of these in System Settings > Privacy & Security:"
echo "  1. Accessibility — for keyboard shortcut detection + text insertion"
echo "  2. Microphone — for voice recording"
echo "  3. Screen Recording — for context-aware transcription (optional)"
echo ""
echo "  If launching from /Applications/, grant permissions to 'VoiceInk'."
echo "  If launching from terminal, grant permissions to your terminal app."
echo ""
echo "Controls: Hold right Option to talk, release to type."
echo "          Double-tap right Option for toggle (hands-free) mode."
echo "          Press Escape to cancel a recording."
