#!/usr/bin/env bash
# 00-baseline run script — runs the full experiment end-to-end.
#
# Usage:
#   bash run.sh                       # run with default config
#   SONFORMER_WANDB=1 bash run.sh     # log to wandb (must `pip install wandb`)
#
# Reproducibility: seed is fixed in config.yaml.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Ensure deps. Skip if you've already installed them globally.
if [[ "${SONFORMER_SKIP_DEPS:-0}" != "1" ]]; then
  pip install -q -r requirements.txt
fi

python train.py
