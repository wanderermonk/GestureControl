#!/usr/bin/env bash
# Start GestureControl. Extra args are passed through, e.g. ./run.sh --dry-run
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run ./setup.sh first."
  exit 1
fi
source .venv/bin/activate
exec python -m gesture_control "$@"
