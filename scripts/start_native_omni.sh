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
OMNI_CTX_SIZE="${IMAGINEOS_OMNI_CTX_SIZE:-131072}"
OMNI_BATCH_SIZE="${IMAGINEOS_OMNI_BATCH_SIZE:-512}"
OMNI_THREADS="${IMAGINEOS_OMNI_THREADS:-8}"
OMNI_REASONING="${IMAGINEOS_OMNI_REASONING:-off}"
# Concurrency/backpressure fix (2026-07-05, docs/HERA_CONCURRENCY_BACKPRESSURE.md):
# infrastructure for pinning --parallel explicitly, DEFAULTED BACK TO AUTO (-1).
#
# INCIDENT (2026-07-05, same session): pinning --parallel 4 here while leaving
# --ctx-size at 32768 silently divides the context pool across slots — per-slot
# usable context collapsed to 32768/4=8192 tokens, and real "standard"/"heavy"
# budget-mode requests (recursive memory + tool schemas routinely ~11K tokens)
# started failing with `exceed_context_size_error`. Caught live in production
# within minutes via a direct MCP smoke-test call. Auto (-1) does NOT divide
# ctx-size the same way (verified: auto picked 4 slots with n_ctx=131072 EACH
# in /props before this incident) — so auto is the known-good state.
#
# Do NOT re-pin --parallel without ALSO raising --ctx-size so that
# ctx-size / OMNI_PARALLEL_SLOTS still leaves an adequate per-slot budget
# (the platform is calibrated for ~131K effective tokens per request — see
# "Context budget calibration" in Hera/CLAUDE.md). hera-core's router.rs
# semaphore (HERA_PRIMARY_ENGINE_SLOTS, default 4) does not depend on this flag
# being pinned — it only needs a reasonable estimate of real concurrent
# capacity, and 4 is what auto already resolves to on this box.
OMNI_PARALLEL_SLOTS="${IMAGINEOS_OMNI_PARALLEL_SLOTS:--1}"

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
#
# KV cache quant default FIXED 2026-07-13 (root cause of the 89-crash OOM
# restart loop, see hera-llm-primary block in ecosystem.config.cjs): this
# script has shipped q8_0/q8_0 since the flag was introduced (commit
# 81c44024, 2026-06-30) -- but Hera/CLAUDE.md's "Context budget calibration"
# note, written the SAME DAY, always documented the design as q4_0/q4_0
# ("KV cache quantized q4_0/q4_0 to fit in 48 GB VRAM"). That was a doc/impl
# mismatch from day one, not a later regression: q4_0 was never actually
# applied. q8_0 KV at --ctx-size 131072 needs ~6528 MiB; q4_0 needs roughly
# half that (measured ~3450-3550 MiB on this box), which is what restores
# enough GPU0 headroom to run the full calibrated 131072 context instead of
# the emergency-patched 65536. Do not silently drop this back to q8_0.
OMNI_KV_CACHE_TYPE_K="${IMAGINEOS_OMNI_KV_CACHE_TYPE_K:-q4_0}"
OMNI_KV_CACHE_TYPE_V="${IMAGINEOS_OMNI_KV_CACHE_TYPE_V:-q4_0}"
OMNI_ROPE_SCALING="${IMAGINEOS_OMNI_ROPE_SCALING:-}"
OMNI_ROPE_SCALE="${IMAGINEOS_OMNI_ROPE_SCALE:-}"
OMNI_YARN_ORIG_CTX="${IMAGINEOS_OMNI_YARN_ORIG_CTX:-}"

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
    --parallel "$OMNI_PARALLEL_SLOTS" \
    --reasoning "$OMNI_REASONING" \
    --flash-attn on \
    --cache-type-k "$OMNI_KV_CACHE_TYPE_K" \
    --cache-type-v "$OMNI_KV_CACHE_TYPE_V"
)

if [ -n "$MMPROJ_FILE" ]; then
    args+=(--mmproj "$MMPROJ_FILE")
fi

# YaRN RoPE scaling — activar cuando se usa Unsloth 128K GGUF
# Setear: IMAGINEOS_OMNI_ROPE_SCALING=yarn IMAGINEOS_OMNI_ROPE_SCALE=4 IMAGINEOS_OMNI_YARN_ORIG_CTX=32768
if [ -n "$OMNI_ROPE_SCALING" ]; then
    args+=(--rope-scaling "$OMNI_ROPE_SCALING")
fi
if [ -n "$OMNI_ROPE_SCALE" ]; then
    args+=(--rope-scale "$OMNI_ROPE_SCALE")
fi
if [ -n "$OMNI_YARN_ORIG_CTX" ]; then
    args+=(--yarn-orig-ctx "$OMNI_YARN_ORIG_CTX")
fi

exec "${args[@]}"
