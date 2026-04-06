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
for f in voice_input.py ner_daemon.swift ner_tool.swift start.sh stop.sh uninstall.sh requirements.txt VERSION VoiceInk.icns icon_light.png icon_dark.png icon_glow.png; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        cp "$SCRIPT_DIR/$f" "$INSTALL_DIR/"
    fi
done
chmod +x "$INSTALL_DIR/start.sh" "$INSTALL_DIR/stop.sh"

# Create default dictionary if not exists
if [[ ! -f "$INSTALL_DIR/dictionary.json" ]]; then
    echo '{"vocabulary": ["Qwen", "MLX", "PyTorch"]}' > "$INSTALL_DIR/dictionary.json"
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
if ! swiftc -O -o "$INSTALL_DIR/ner_tool" "$INSTALL_DIR/ner_tool.swift"; then
    echo "Error: Failed to compile ner_tool.swift"
    exit 1
fi
if ! swiftc -O -o "$INSTALL_DIR/ner_daemon" "$INSTALL_DIR/ner_daemon.swift"; then
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
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/voice_input.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/voice_input.log</string>
</dict>
</plist>
PLISTEOF

# Copy .app to /Applications (optional — may need sudo)
echo "Installing VoiceInk.app..."
if [[ -d "$SCRIPT_DIR/VoiceInk.app" ]]; then
    sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true
    sudo cp -R "$SCRIPT_DIR/VoiceInk.app" /Applications/ 2>/dev/null || echo "  Skipped (no sudo). You can manually copy VoiceInk.app to /Applications/."
elif [[ -d "$INSTALL_DIR/VoiceInk.app" ]]; then
    sudo rm -rf /Applications/VoiceInk.app 2>/dev/null || true
    sudo cp -R "$INSTALL_DIR/VoiceInk.app" /Applications/ 2>/dev/null || echo "  Skipped (no sudo)."
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Start VoiceInk:"
echo "  Option 1: Click VoiceInk in /Applications/"
echo "  Option 2: ~/.local/voice-input/start.sh"
echo ""
echo "First launch will download the ASR model (~3.4 GB)."
echo "You will need to grant these permissions in System Settings > Privacy & Security:"
echo "  1. Accessibility — for keyboard shortcut detection + text pasting"
echo "  2. Microphone — for voice recording"
echo "  3. Screen Recording — for context-aware transcription (optional)"
echo ""
echo "Controls: Hold right Option to talk, release to type."
echo "          Double-tap right Option for toggle (hands-free) mode."
echo "          Press Escape to cancel a recording."
