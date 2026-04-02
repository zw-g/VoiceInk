#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv-py2app/bin/python" "$DIR/voice_input.py" "$@"
