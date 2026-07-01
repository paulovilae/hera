#!/bin/bash
# Native Omni Model Launcher
# ImagineOS - VRAM Unified Optimization

echo "🚀 Starting ImagineOS Native Omni Engine"

MODEL_DIR="/mnt/workspace/data/models/llm-stack"
MODEL_BIN="/home/paulo/llama.cpp/build/bin/llama-server"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"

MODEL_FILE="${IMAGINEOS_OMNI_MODEL_FILE:-$MODEL_DIR/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
MMPROJ_FILE="${IMAGINEOS_OMNI_MMPROJ_FILE:-}"
OMNI_PORT="${IMAGINEOS_OMNI_PORT:-8080}"
OMNI_HOST="${IMAGINEOS_OMNI_HOST:-127.0.0.1}"
OMNI_GPU_LAYERS="${IMAGINEOS_OMNI_GPU_LAYERS:-99}"
OMNI_MAIN_GPU="${IMAGINEOS_OMNI_MAIN_GPU:-0}"
OMNI_CTX_SIZE="${IMAGINEOS_OMNI_CTX_SIZE:-32768}"
OMNI_BATCH_SIZE="${IMAGINEOS_OMNI_BATCH_SIZE:-512}"
OMNI_THREADS="${IMAGINEOS_OMNI_THREADS:-8}"
OMNI_REASONING="${IMAGINEOS_OMNI_REASONING:-off}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while [ ! -f "$MODEL_FILE" ] || { [ -n "$MMPROJ_FILE" ] && [ ! -f "$MMPROJ_FILE" ]; }; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "❌ Error: Required GGUF model assets not found in $MODEL_DIR after ${WAIT_TIMEOUT_SECONDS}s"
        exit 1
    fi

    echo "⏳ Waiting for model assets in $MODEL_DIR (retry in ${WAIT_INTERVAL_SECONDS}s)"
    [ -f "$MODEL_FILE" ] || echo "   missing model: $MODEL_FILE"
    [ -z "$MMPROJ_FILE" ] || [ -f "$MMPROJ_FILE" ] || echo "   missing projector: $MMPROJ_FILE"
    sleep "$WAIT_INTERVAL_SECONDS"
done

echo "🧠 Model: $MODEL_FILE"
if [ -n "$MMPROJ_FILE" ]; then
    echo "👁️ Projector: $MMPROJ_FILE"
else
    echo "👁️ Projector: disabled (text-only)"
fi

# Unload existing GPUs (Rule #7)
echo "🧹 Releasing VRAM before native binding..."
# pm2 stop all text LLMs if exist is handled by user, but we'll run memory clear

# Start Native Llama Server.
# --split-mode none + --main-gpu 0 = Force ALL layers onto GPU 0 (24GB)
# This keeps GPU 1 completely free for FLUX (12GB) + Wan2.1 video
args=(
    "$MODEL_BIN"
    --model "$MODEL_FILE" \
    --port "$OMNI_PORT" \
    --host "$OMNI_HOST" \
    --gpu-layers "$OMNI_GPU_LAYERS" \
    --split-mode none \
    --main-gpu "$OMNI_MAIN_GPU" \
    --ctx-size "$OMNI_CTX_SIZE" \
    --batch-size "$OMNI_BATCH_SIZE" \
    --threads "$OMNI_THREADS" \
    --reasoning "$OMNI_REASONING"
)

if [ -n "$MMPROJ_FILE" ]; then
    args+=(--mmproj "$MMPROJ_FILE")
fi

exec "${args[@]}"
