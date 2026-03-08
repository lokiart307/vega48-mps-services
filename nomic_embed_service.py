#!/usr/bin/env python3
"""
Nomic embed text v2 moe — embedding endpoint
POST /embed  {"text": "..."}  →  {"embedding": [...768 floats...]}
POST /embed/batch  {"texts": ["...", "..."]}  →  {"embeddings": [[...], [...]]}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn
import torch

MODEL_NAME = "nomic-ai/nomic-embed-text-v2-moe"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

app = FastAPI()
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)


class EmbedRequest(BaseModel):
    text: str

class BatchEmbedRequest(BaseModel):
    texts: List[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    emb = model.encode([req.text], normalize_embeddings=True)[0]
    return {"embedding": emb.tolist()}


@app.post("/embed/batch")
def embed_batch(req: BatchEmbedRequest):
    embs = model.encode(req.texts, normalize_embeddings=True, batch_size=128)
    return {"embeddings": [e.tolist() for e in embs]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "dim": 768}


if __name__ == "__main__":
    uvicorn.run(app, host="100.69.11.31", port=8021)
