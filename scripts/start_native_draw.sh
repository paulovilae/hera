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
    --seed -1 \
    --lora-model-dir /home/paulo/models/image-stack/loras \
    --fa

# --seed -1  → fresh RANDOM seed per generation (diversity fix, 2026-07-13).
# Root cause of "same prompt = identical image every time": this sd-server build's
# OpenAI-compat endpoint (/v1/images/generations) IGNORES the per-request `seed`
# (and `steps`/`cfg_scale`) in the JSON body — it only honors the process-level
# flags here, and the default seed is a FIXED 42 (confirmed in the sd-server log:
# "generating image: 1/1 - seed 42" for every request, regardless of the seed sent
# by Hera/Imaginclaw). With the default fixed seed, an identical prompt produced
# byte-identical output on every call. `--seed -1` makes the sampler draw a random
# seed each time, restoring variety. TRADE-OFF: because the endpoint ignores the
# per-request seed, the "seed:N" reproducibility feature in Imaginclaw's /draw
# caption cannot actually reproduce a specific image (it never could — the server
# was always using 42); it is now non-deterministic. Fixing true per-request seed
# control would require rebuilding sd.cpp with an endpoint that parses `seed`
# (source not present on the node — only build/ is checked out), out of scope here.
# steps=8/cfg=1.0 are kept: Z-Image-Turbo is distilled for ~8 few-steps at low CFG;
# raising them risks over-cooking, and the reported symptom was the fixed seed, not
# the step count.
