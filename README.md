<p align="center">
  <img src="assets/icon_light.png" alt="VoiceInk" width="128" height="128">
</p>

<h1 align="center">VoiceInk</h1>

<p align="center">
  macOS menu bar push-to-talk voice input powered by <b>Qwen3-ASR</b> on Apple Silicon.<br>
  Press right Option to talk, release to type. That's it.
</p>

<p align="center">
  <img src="assets/icon_light.png" alt="Light" width="64" height="64">
  <img src="assets/icon_dark.png" alt="Dark" width="64" height="64">
  <img src="assets/icon_glow.png" alt="Glow" width="64" height="64">
</p>

## Features

- **Push-to-talk** (hold right Option) or **toggle mode** (double-tap right Option)
- **Qwen3-ASR 1.7B** — one of the strongest open-source ASR models, running locally on your Mac
- **Screen context** — OCR captures all visible screens, NER extracts proper nouns, feeds them as context to ASR for better accuracy
- **Custom dictionary** — add domain-specific terms for better recognition
- **Smart microphone management** — auto-detects device changes, remembers your preference
- **Native SF Symbol animations** — waveform icon with macOS-native effects
- **Transcription history** — last 30 transcriptions saved, click to copy
- **Escape to cancel** — press Escape to discard a recording
- **Privacy first** — everything runs locally. No data leaves your Mac.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14+ (Sonoma or newer)
- Python 3.13 (`brew install python@3.13`)
- ffmpeg (`brew install ffmpeg`) — for non-WAV audio formats

## Install

```bash
git clone https://github.com/zw-g/VoiceInk.git
cd VoiceInk
./install.sh
```

The install script will:
1. Create a Python virtual environment
2. Install all dependencies (mlx-qwen3-asr, sounddevice, pynput, rumps, etc.)
3. Compile the Swift NER tool
4. Set up the LaunchAgent
5. Copy VoiceInk.app to /Applications/

## Usage

### Start
```bash
# Option 1: Click VoiceInk in /Applications/
# Option 2: From terminal
~/.local/voice-input/start.sh
```

### Controls
| Action | Key |
|---|---|
| Push-to-talk | Hold right Option, release to transcribe |
| Toggle recording | Double-tap right Option |
| Cancel recording | Press Escape |
| Stop toggle recording | Tap right Option again |

### Menu Bar
Click the waveform icon (top-right) to access:
- Recent Transcriptions (click to copy)
- Microphone selection
- Screen Context toggle
- Edit Dictionary
- Launch at Login
- Quit

### Configuration

Edit `~/.local/voice-input/settings.json`:
```json
{
  "preferred_mic": "MacBook Pro Microphone",
  "screen_context": true,
  "hotkey": "alt_r",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "ocr_languages": ["en", "zh-Hans", "zh-Hant"]
}
```

Hotkey options: `alt_r`, `alt_l`, `cmd_r`, `ctrl_r`, `f18`, `f19`, `f20`

### Custom Dictionary

Edit `~/.local/voice-input/dictionary.json`:
```json
{
  "vocabulary": ["Qwen", "MLX", "PyTorch", "Phabricator", "Claude"]
}
```
Changes auto-reload within 2 seconds.

## Permissions

On first launch, VoiceInk will check for required permissions:
- **Accessibility** — for keyboard monitoring + text pasting
- **Microphone** — for audio recording
- **Screen Recording** — for screen context OCR

## How It Works

```
You press Option → Record audio → OCR all screens → NER extracts proper nouns
                                                    ↓
Release Option → ASR transcribes with context → Paste text at cursor
```

1. Audio recorded via sounddevice
2. All visible screens captured via CGWindowListCreateImage
3. Vision framework OCR extracts text
4. Apple NaturalLanguage NER extracts proper nouns and technical terms
5. Dictionary + NER terms passed as `context` to Qwen3-ASR decoder
6. Transcribed text pasted at cursor via NSPasteboard + Cmd+V

## Architecture

- **ASR**: Qwen3-ASR 1.7B via [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr)
- **OCR**: Apple Vision framework (VNRecognizeTextRequest)
- **NER**: Apple NaturalLanguage framework (compiled Swift daemon)
- **UI**: rumps (NSStatusBar menu bar app)
- **Keyboard**: pynput (Quartz event tap)
- **Audio**: sounddevice (PortAudio)

## System Requirements

- **CPU**: Apple Silicon (M1/M2/M3/M4) — required for MLX
- **macOS**: 14 (Sonoma) or newer
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: ~4 GB for ASR model download
- **Homebrew**: required for Python 3.13 and ffmpeg

## Troubleshooting

### Nothing happens when I press right Option
1. Check Accessibility permission: **System Settings > Privacy & Security > Accessibility** — make sure the Python process or VoiceInk is listed and enabled
2. Check the log: `tail -20 ~/.local/voice-input/voice_input.log`
3. The model may still be loading — wait for the "Ready" notification

### Text doesn't appear after recording
1. Check Accessibility permission (needed for Cmd+V simulation)
2. Make sure the cursor is in a text field before recording
3. Try a longer recording (very short recordings < 0.3s are skipped)

### Model download is stuck
The first launch downloads ~3.4 GB. Check your internet connection and try again:
```bash
pkill -f voice_input.py
~/.local/voice-input/start.sh
```

### Menu bar icon not visible
The icon is a small waveform `〰️` in the top-right menu bar area. It may be hidden behind other icons. Try clicking near the clock area.

### Screen context not working
Grant Screen Recording permission: **System Settings > Privacy & Security > Screen Recording**

### Wrong microphone selected
Click the menu bar icon > Microphone > select your preferred device. The preference is saved across restarts.

### App crashes or freezes
Check the log for errors:
```bash
tail -50 ~/.local/voice-input/voice_input.log
```
The app auto-restarts on crashes (via LaunchAgent KeepAlive). If issues persist, try reinstalling:
```bash
cd /path/to/VoiceInk
./install.sh
```

## Uninstall

```bash
# Stop the app
pkill -f voice_input.py
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.local.voiceinput.plist 2>/dev/null

# Remove files
rm -rf ~/.local/voice-input
rm -f ~/Library/LaunchAgents/com.local.voiceinput.plist
sudo rm -rf /Applications/VoiceInk.app
```

## License

MIT
