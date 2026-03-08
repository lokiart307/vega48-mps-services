#!/usr/bin/env python3
"""
SDXL base image generation endpoint
Radeon Pro Vega 48 via MPS

POST /generate  {"prompt": "...", "negative_prompt": "...", "steps": 20, "width": 1024, "height": 1024}
GET  /health
"""

import base64, io, torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

print(f"Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16",
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print("SDXL loaded.")

app = FastAPI(title="SDXL Service")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, bad quality, watermark"
    steps: Optional[int] = 20
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    guidance_scale: Optional[float] = 7.5


@app.post("/generate")
def generate(req: GenerateRequest):
    with torch.inference_mode():
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
        )
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": b64, "width": img.width, "height": img.height}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE)}


if __name__ == "__main__":
    uvicorn.run(app, host="100.69.11.31", port=8023)
