#!/usr/bin/env bash
# Double-click in Finder to start GestureControl in Terminal.
cd "$(dirname "$0")"
[ -d .venv ] || ./setup.sh
./run.sh
