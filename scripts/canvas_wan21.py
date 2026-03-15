#!/usr/bin/env python3
"""
ImagineOS Native Video Canvas — Wan2.1 T2V 1.3B
Sovereign bare-metal video generation server.
Binds to 127.0.0.1:8091 on GPU 1.
"""

import os
import sys
import time
import uuid
import json
import torch
import imageio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Config ───────────────────────────────────────────────────────────
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CACHE_DIR = "/data/models/wan2.1"
OUTPUT_DIR = "/tmp/imagineos-canvas"
PORT = 8091
GPU_DEVICE = 1  # Physical GPU 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
# Reduce VRAM fragmentation when sharing with FLUX sd.cpp
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="ImagineOS Canvas — Wan2.1", version="1.0.0")


class VideoRequest(BaseModel):
    prompt: str
    width: int = 480
    height: int = 320
    num_frames: int = 33
    num_inference_steps: int = 20
    guidance_scale: float = 5.0


class VideoResponse(BaseModel):
    status: str
    path: str
    duration_seconds: float


# Global pipeline (loaded at startup)
pipe = None


def load_pipeline():
    """Load Wan2.1 T2V pipeline with CPU offloading (VRAM-friendly)."""
    global pipe
    from diffusers import WanPipeline

    print(f"📥 Loading Wan2.1 T2V 1.3B from {MODEL_ID}...")
    print(f"   Cache dir: {CACHE_DIR}")
    print(f"   Target GPU: cuda:0 (mapped from physical GPU {GPU_DEVICE})")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    # CRITICAL: Use enable_model_cpu_offload WITHOUT calling .to("cuda")
    # This keeps model weights in CPU RAM and only moves layers to GPU
    # one at a time during the forward pass, minimizing peak VRAM usage.
    pipe.enable_model_cpu_offload(gpu_id=0)

    print("✅ Wan2.1 pipeline loaded and ready! (CPU-offload mode)")


@app.on_event("startup")
async def startup_event():
    load_pipeline()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": GPU_DEVICE}


@app.post("/v1/video/generate", response_model=VideoResponse)
async def generate_video(req: VideoRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    start = time.time()
    video_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"video_{video_id}.mp4")

    try:
        print(f"🎬 Generating video: '{req.prompt[:80]}...'")
        print(f"   Resolution: {req.width}x{req.height}, frames: {req.num_frames}, steps: {req.num_inference_steps}")
        output = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        )

        # Export frames to MP4
        frames = output.frames[0]  # List of PIL images
        writer = imageio.get_writer(output_path, fps=16, codec="libx264")
        for frame in frames:
            import numpy as np
            writer.append_data(np.array(frame))
        writer.close()

        elapsed = time.time() - start
        print(f"✅ Video saved: {output_path} ({elapsed:.1f}s)")

        # Free GPU memory after generation to coexist with FLUX
        torch.cuda.empty_cache()

        return VideoResponse(
            status="success",
            path=output_path,
            duration_seconds=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Generation failed after {elapsed:.1f}s: {e}")
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
