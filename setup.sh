#!/usr/bin/env bash
# One-time setup: creates .venv, installs dependencies, downloads the hand model.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    case "$ver" in 3.10|3.11|3.12|3.13) PY="$cand"; break ;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "Need Python 3.10-3.13 (MediaPipe doesn't support other versions yet)."
  echo "Install one with:  brew install python@3.12"
  exit 1
fi
echo "Using $PY ($($PY --version))"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# Ignore any global pip config (e.g. an extra NVIDIA index) and use plain PyPI.
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
PIP_OPTS=(--index-url https://pypi.org/simple --disable-pip-version-check)
python -m pip install "${PIP_OPTS[@]}" --upgrade pip -q
python -m pip install "${PIP_OPTS[@]}" -r requirements.txt

mkdir -p models
if [ ! -s models/hand_landmarker.task ]; then
  echo "Downloading hand landmark model (~7.5 MB)..."
  curl -fL --progress-bar -o models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi

# Voice models (offline speech recognition + speaker ID), ~55 MB total
VOICE_MODEL=$(python -c "from gesture_control.config import load_config; print(load_config()['voice']['model'])")
for m in "$VOICE_MODEL" vosk-model-spk-0.4; do
  if [ ! -d "models/$m" ]; then
    echo "Downloading voice model $m ..."
    curl -fL --progress-bar -o "models/$m.zip" "https://alphacephei.com/vosk/models/$m.zip"
    unzip -q -o "models/$m.zip" -d models && rm "models/$m.zip"
  fi
done

echo "Running self-tests..."
python -m pytest -q tests

chmod +x run.sh "Start GestureControl.command" 2>/dev/null || true
cat <<'EOF'

Setup complete.
Start it with:   ./run.sh            (or double-click "Start GestureControl.command")
Try safely:      ./run.sh --dry-run  (shows gestures, doesn't move the mouse)

Voice:           ./run.sh --train-voice   (teach it your voice, ~4 min)
                 ./run.sh --voice-test    (see what it hears)

The first run will ask for Camera, Microphone and Accessibility permission for your terminal app.
EOF
