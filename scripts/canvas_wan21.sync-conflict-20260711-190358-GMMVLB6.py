#!/usr/bin/env python3
"""
ImagineOS Native Video Canvas — Wan2.1 VACE 1.3B + T2V 1.3B
Sovereign bare-metal video generation server.
Supports both Text-to-Video AND Image-to-Video (FLUX anchor frame pipeline).
Binds to 127.0.0.1:8091 on GPU 1.
"""

import os
import sys
import time
import uuid
import json
import base64
import io
import torch
import imageio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Config ───────────────────────────────────────────────────────────
T2V_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-Diffusers"
T2V_CACHE_DIR = "/data/models/wan2.1"
VACE_CACHE_DIR = "/data/models/wan2.1-vace"
OUTPUT_DIR = "/tmp/imagineos-canvas"
PORT = int(os.environ.get("CANVAS_PORT", "8092"))
GPU_DEVICE = 1  # Physical GPU 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="ImagineOS Canvas — Wan2.1", version="2.0.0")


class VideoRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None  # Base64-encoded anchor image for I2V
    width: int = 480
    height: int = 320
    num_frames: int = 81
    num_inference_steps: int = 20
    guidance_scale: float = 5.0


# Global pipelines (loaded at startup)
t2v_pipe = None
vace_pipe = None


def load_pipelines():
    """Load T2V and optionally VACE pipelines with CPU offloading."""
    global t2v_pipe, vace_pipe

    # Load T2V (always available as fallback)
    from diffusers import WanPipeline
    print(f"📥 Loading Wan2.1 T2V 1.3B from {T2V_MODEL_ID}...")
    t2v_pipe = WanPipeline.from_pretrained(
        T2V_MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=T2V_CACHE_DIR,
    )
    t2v_pipe.enable_model_cpu_offload(gpu_id=0)
    print("✅ T2V pipeline loaded!")

    # Load VACE I2V (if model is downloaded)
    vace_path = Path(VACE_CACHE_DIR)
    if vace_path.exists() and any(vace_path.iterdir()):
        try:
            from diffusers import WanVACEPipeline
            print(f"📥 Loading Wan2.1 VACE I2V from {VACE_MODEL_ID}...")
            vace_pipe = WanVACEPipeline.from_pretrained(
                VACE_MODEL_ID,
                torch_dtype=torch.float16,
                cache_dir=VACE_CACHE_DIR,
            )
            vace_pipe.enable_model_cpu_offload(gpu_id=0)
            print("✅ VACE I2V pipeline loaded!")
        except Exception as e:
            print(f"⚠️ VACE I2V failed to load: {e}. Falling back to T2V only.")
            vace_pipe = None
    else:
        print("ℹ️ VACE model not found. Running T2V only.")


@app.on_event("startup")
async def startup_event():
    load_pipelines()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "t2v_model": T2V_MODEL_ID,
        "vace_model": VACE_MODEL_ID if vace_pipe else "not loaded",
        "gpu": GPU_DEVICE,
        "i2v_capable": vace_pipe is not None,
    }


@app.post("/v1/video/generate")
async def generate_video(req: VideoRequest):
    """Generate video from text, or from image+text if image_base64 is provided."""
    from PIL import Image
    import numpy as np

    # Decide which pipeline to use
    use_i2v = req.image_base64 is not None and vace_pipe is not None
    active_pipe = vace_pipe if use_i2v else t2v_pipe

    if active_pipe is None:
        raise HTTPException(status_code=503, detail="No pipeline loaded")

    start = time.time()
    video_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"video_{video_id}.mp4")

    try:
        mode = "I2V (VACE)" if use_i2v else "T2V"
        print(f"🎬 [{mode}] Generating video: '{req.prompt[:80]}...'")
        print(f"   Resolution: {req.width}x{req.height}, frames: {req.num_frames}, steps: {req.num_inference_steps}")

        if use_i2v:
            # Decode base64 image
            img_bytes = base64.b64decode(req.image_base64)
            anchor_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Preserve aspect ratio: fit within max 480px, round to 16px
            orig_w, orig_h = anchor_image.size
            max_dim = 480
            scale = min(max_dim / orig_w, max_dim / orig_h)
            new_w = int(orig_w * scale) // 16 * 16
            new_h = int(orig_h * scale) // 16 * 16
            new_w = max(new_w, 128)
            new_h = max(new_h, 128)
            anchor_image = anchor_image.resize((new_w, new_h))
            # Override request dimensions to match
            req.width = new_w
            req.height = new_h
            print(f"   🖼️ Anchor image: {orig_w}x{orig_h} → {new_w}x{new_h}")

            # Build video tensor: first frame = user image, rest = black
            black_frame = Image.new("RGB", (req.width, req.height), (0, 0, 0))
            video_frames = [anchor_image] + [black_frame] * (req.num_frames - 1)

            # Build mask: VACE convention — 0=keep, 255=generate
            # First frame = black (keep user image), rest = white (generate motion)
            keep_mask = Image.new("L", (req.width, req.height), 0)      # keep
            gen_mask = Image.new("L", (req.width, req.height), 255)     # generate
            mask_frames = [keep_mask] + [gen_mask] * (req.num_frames - 1)

            output = active_pipe(
                prompt=req.prompt,
                video=video_frames,
                mask=mask_frames,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
            )
        else:
            output = active_pipe(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
            )

        # Export frames to MP4
        frames = output.frames[0]
        writer = imageio.get_writer(output_path, fps=16, codec="libx264")
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        elapsed = time.time() - start
        print(f"✅ Video saved: {output_path} ({elapsed:.1f}s)")
        torch.cuda.empty_cache()

        return JSONResponse(content={
            "status": "success",
            "path": output_path,
            "duration_seconds": round(elapsed, 2),
            "mode": mode,
        })

    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Generation failed after {elapsed:.1f}s: {e}")
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
