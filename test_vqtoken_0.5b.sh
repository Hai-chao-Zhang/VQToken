#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

# The released VQToken checkpoint is currently gated. The default public base
# checkpoint still exercises this repository's VQToken compression path without
# requiring an API key or downloading the 120+ GiB ActivityNetQA benchmark.
PRETRAIN=${PRETRAIN:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
REVISION=${REVISION:-}
VIDEO=${VIDEO:-$REPO_ROOT/playground/demo/xU25MMA2N4aVtYay.mp4}
DEVICE=${DEVICE:-cuda:0}
FRAMES=${FRAMES:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
SELECTION=${SELECTION:-fixed}

ARGS=(
    --pretrained "$PRETRAIN"
    --video "$VIDEO"
    --device "$DEVICE"
    --frames "$FRAMES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --selection "$SELECTION"
)
if [[ -n "$REVISION" ]]; then
    ARGS+=(--revision "$REVISION")
fi
ARGS+=("$@")

python scripts/smoke_inference.py "${ARGS[@]}"
