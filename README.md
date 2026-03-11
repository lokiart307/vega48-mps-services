# Vega 48 MPS Services

ML inference on a 2019 iMac using PyTorch MPS on the AMD discrete GPU.

Every LLM and most online resources say MPS only works on Apple Silicon, or that you need ROCm or CUDA for GPU inference on AMD hardware. That's not true — PyTorch's MPS backend works via Metal on Intel Macs with AMD discrete GPUs. It might be a recent addition to PyTorch, or it might have always worked and just isn't well documented. Either way, it works.

I kept being told it wasn't possible, tried it anyway, and was happy to find out it was. If you've got one of these machines gathering dust, maybe this saves you some time.

## Hardware

- 2019 iMac (iMac19,1)
- 8-core Intel i9 3.6GHz
- 80GB RAM
- Radeon Pro Vega 48, 8GB VRAM
- macOS Sonoma 14.6.1

## Services

### Text Embeddings — `nomic_embed_service.py`

FastAPI service running [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) on MPS. 768-dimensional normalised embeddings.

- `POST /embed` — single text → 768-dim embedding
- `POST /embed/batch` — list of texts, batch_size=128 internally
- `GET /health`
- Port 8021

**Benchmarks:**

| Mode | Throughput |
|------|-----------|
| Sequential single | 3.4 embeds/sec (~293ms/req) |
| Batch 128 (sweet spot) | 28.3 embeds/sec (35ms/embed) |
| Batch 256 | 25.4 embeds/sec (VRAM pressure) |
| 8 workers × batch 32 | 59.1 embeds/sec |
| 4 workers × batch 128 | 102.5 embeds/sec |
| 8 workers × batch 128 | 111.8 embeds/sec (peak) |

Batch 128 is the sweet spot. 256 drops off — likely VRAM pressure on the 8GB card. 4 workers × batch 128 is the efficient operating point at ~103 embeds/sec; going to 8 workers only gains ~9% more.

### Vision Captioning — `moondream_service.py`

[Moondream2](https://huggingface.co/vikhyatk/moondream2) (1.8B) in fp16 on MPS. Captioning, visual Q&A, batch captioning.

- `POST /caption` — base64 image → caption (short/normal/long)
- `POST /query` — base64 image + question → answer
- `POST /caption/batch` — list of base64 images → captions
- `GET /health`
- Port 8022
- ~3.5GB VRAM

**Benchmarks:**

| Mode | Speed |
|------|-------|
| Short captions (seq) | 12.3s/cap, 0.08 cap/sec |
| Normal captions (seq) | 28.8s/cap, 0.03 cap/sec |
| 2-4 workers | No throughput gain — MPS serializes |
| Batch 2-8 | Sequential internally |
| RAM | Stable ~20.5GB, ~44GB free |

Concurrency does not help. MPS serializes all GPU work.

### Image Generation — `sdxl_service.py`

[SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) in fp16 on MPS with DPMSolverMultistepScheduler, attention slicing, and VAE slicing.

- `POST /generate` — prompt → base64 PNG (configurable steps, size, guidance, negative prompt)
- `GET /health`
- Port 8023
- ~6.5GB VRAM

Uses DPMSolverMultistepScheduler because the default Euler scheduler has an off-by-one IndexError in diffusers 0.30.x.

### Queue Runner — `sdxl_queue.py`

Pulls prompts from the Mac Studio over SSH, generates via the SDXL service, saves PNGs locally. Resumable — logs to `queue_log.json`, skips completed items on restart. Truncates prompts to 60 words for CLIP's 77-token limit.

## VRAM

The Vega 48 has 8GB. Embeddings + captioning fit together (~5.5GB). SDXL needs ~6.5GB — stop other services first.

## Setup

```bash
pip install -r requirements.txt

python nomic_embed_service.py
uvicorn moondream_service:app --host 0.0.0.0 --port 8022 --workers 1
uvicorn sdxl_service:app --host 0.0.0.0 --port 8023 --workers 1
```

## Notes

- Services bind to Tailscale IP (100.69.11.31), not localhost
- Moondream 2025-01-09 revision needs `from __future__ import annotations` in `image_crops.py` (Python 3.9+ syntax)
- Should work on any Intel Mac with AMD discrete GPU (Vega 56/64, 5500 XT, 5700 XT, W5700X, etc.)

## License

MIT
