#!/usr/bin/env bash
# Sovereign ImagineOS Native Draw Engine Wrapper
# C++ sd.cpp server. Modelo: Z-Image-Turbo (mejor anatomía y texto que
# FLUX-schnell — arregla los headless/manos deformes; usa encoder Qwen3-4B vía
# --llm en vez de clip_l/t5xxl). El comando FLUX previo queda respaldado en
# start_native_draw.flux.bak para revertir en 1 minuto.

# Default ENV values if Argus orchestration isn't explicitly providing them
GPU_TARGET="${DRAW_ENGINE_CUDA_DEVICE:-1}"
LISTEN_PORT="${DRAW_ENGINE_PORT:-8999}"
LISTEN_IP="${DRAW_ENGINE_LISTEN_IP:-127.0.0.1}"

echo "Starting ImagineOS Native Draw Engine (Z-Image-Turbo) on GPU ${GPU_TARGET} at port ${LISTEN_PORT}..."

# Export the targeted GPU so the C++ backend maps it to 'Device 0'
export CUDA_VISIBLE_DEVICES="${GPU_TARGET}"

/home/paulo/sd.cpp/build/bin/sd-server \
    --diffusion-model /home/paulo/models/image-stack/zimage/z_image_turbo-Q6_K.gguf \
    --vae /home/paulo/.cache/huggingface/hub/models--receptektas--black-forest-labs-ae_safetensors/snapshots/a45f46d48f133e6711d89447d8c8601d1939c9e1/ae.safetensors \
    --llm /home/paulo/models/image-stack/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --listen-port "${LISTEN_PORT}" \
    --listen-ip "${LISTEN_IP}" \
    --steps 8 \
    --cfg-scale 1.0 \
    --lora-model-dir /home/paulo/models/image-stack/loras \
    --fa \
    --vae-on-cpu
