#!/usr/bin/env bash
# 02a-rope run script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [[ "${SONFORMER_SKIP_DEPS:-0}" != "1" ]]; then
  pip install -q -r requirements.txt
fi
python train.py
