#!/usr/bin/env python3
"""
Moondream2 vision-language model — inference endpoint
Running on Radeon Pro Vega 48 via MPS

POST /caption  {"image": "<base64>"}  →  {"caption": "..."}
POST /query    {"image": "<base64>", "question": "..."}  →  {"answer": "..."}
POST /caption/batch  {"images": ["<base64>", ...]}  →  {"captions": [...]}
GET  /health   →  {"status": "ok", "model": "...", "device": "mps"}
"""

import base64
import io
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from transformers import AutoModelForCausalLM
import uvicorn

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-01-09"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

print(f"Loading {MODEL_ID} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).to(DEVICE)
model.eval()
print("Model loaded.")

app = FastAPI(title="Moondream2 Service")


def decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


class CaptionRequest(BaseModel):
    image: str  # base64
    length: Optional[str] = "normal"  # "short" | "normal" | "long"

class QueryRequest(BaseModel):
    image: str  # base64
    question: str

class BatchCaptionRequest(BaseModel):
    images: List[str]  # base64 list
    length: Optional[str] = "normal"


@app.post("/caption")
def caption(req: CaptionRequest):
    img = decode_image(req.image)
    with torch.inference_mode():
        enc = model.encode_image(img)
        result = model.caption(enc, length=req.length)
    return {"caption": result["caption"] if isinstance(result, dict) else result}


@app.post("/query")
def query(req: QueryRequest):
    img = decode_image(req.image)
    with torch.inference_mode():
        enc = model.encode_image(img)
        result = model.query(enc, req.question)
    return {"answer": result["answer"] if isinstance(result, dict) else result}


@app.post("/caption/batch")
def caption_batch(req: BatchCaptionRequest):
    captions = []
    for b64 in req.images:
        img = decode_image(b64)
        with torch.inference_mode():
            enc = model.encode_image(img)
            result = model.caption(enc, length=req.length)
        captions.append(result["caption"] if isinstance(result, dict) else result)
    return {"captions": captions}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "revision": REVISION,
        "device": DEVICE,
        "dtype": str(DTYPE),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="100.69.11.31", port=8022)
