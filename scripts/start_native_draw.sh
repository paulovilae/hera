#!/usr/bin/env bash
# Sovereign ImagineOS Native Draw Engine Wrapper
# This script loads environment variables and bootstraps the C++ sd.cpp server
# ensuring optimal multi-GPU topology separation.

# Default ENV values if Argus orchestration isn't explicitly providing them
GPU_TARGET="${DRAW_ENGINE_CUDA_DEVICE:-1}"
LISTEN_PORT="${DRAW_ENGINE_PORT:-8999}"
LISTEN_IP="${DRAW_ENGINE_LISTEN_IP:-0.0.0.0}"

echo "Starting ImagineOS Native Draw Engine on GPU ${GPU_TARGET} at port ${LISTEN_PORT}..."

# Export the targeted GPU so the C++ backend maps it to 'Device 0'
export CUDA_VISIBLE_DEVICES="${GPU_TARGET}"

/home/paulo/sd.cpp/build/bin/sd-server \
    --diffusion-model /home/paulo/models/image-stack/flux1-dev-Q8_0.gguf \
    --vae /home/paulo/.cache/huggingface/hub/models--receptektas--black-forest-labs-ae_safetensors/snapshots/a45f46d48f133e6711d89447d8c8601d1939c9e1/ae.safetensors \
    --clip_l /home/paulo/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors \
    --t5xxl /home/paulo/.cache/huggingface/hub/models--city96--t5-v1_1-xxl-encoder-gguf/snapshots/005a6ea51a7d0b84d677b3e633bb52a8c85a83d9/t5-v1_1-xxl-encoder-Q8_0.gguf \
    --listen-port "${LISTEN_PORT}" \
    --listen-ip "${LISTEN_IP}" \
    --steps 20 \
    --cfg-scale 1.0 \
    --lora-model-dir /home/paulo/models/image-stack/loras/ \
    --fa
