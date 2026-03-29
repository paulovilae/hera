#!/bin/bash
# Native Omni Model Launcher
# ImagineOS - VRAM Unified Optimization

echo "🚀 Starting ImagineOS Native Omni Engine (Qwen3-VL 30B)"

MODEL_DIR="/data/models/llm-stack"
MODEL_BIN="/home/paulo/llama.cpp/build/bin/llama-server"

# Identify specific Qwen3-VL and Projector
MODEL_FILE="$MODEL_DIR/Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
MMPROJ_FILE="$MODEL_DIR/Qwen3-VL-mmproj-F16.gguf"

if [ ! -f "$MODEL_FILE" ] || [ ! -f "$MMPROJ_FILE" ]; then
    echo "❌ Error: Required GGUF model or mmproj projector not found in $MODEL_DIR"
    exit 1
fi

echo "🧠 Model: $MODEL_FILE"
echo "👁️ Projector: $MMPROJ_FILE"

# Unload existing GPUs (Rule #7)
echo "🧹 Releasing VRAM before native binding..."
# pm2 stop all text LLMs if exist is handled by user, but we'll run memory clear

# Start Native Llama Server on Port 8080 with Multi-modal capabilities
# --split-mode none + --main-gpu 0 = Force ALL layers onto GPU 0 (24GB)
# This keeps GPU 1 completely free for FLUX (12GB) + Wan2.1 video
exec "$MODEL_BIN" \
    --model "$MODEL_FILE" \
    --mmproj "$MMPROJ_FILE" \
    --port 8080 \
    --host 127.0.0.1 \
    --gpu-layers 99 \
    --split-mode none \
    --main-gpu 0 \
    --ctx-size 32768 \
    --batch-size 512 \
    --threads 8
