# Vega 48 MPS Services

Three ML services running on PyTorch MPS on an Intel Mac with an AMD discrete GPU.

## Hardware

- 2019 iMac (iMac19,1) — 8-core Intel i9 3.6GHz, 80GB RAM
- Radeon Pro Vega 48, 8GB VRAM
- macOS Sonoma 14.6.1

## Services

### Text Embeddings — `nomic_embed_service.py`

FastAPI service running [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) on MPS. 768-dimensional normalised embeddings. Single and batch endpoints.

- `POST /embed` — single text → 768-dim embedding
- `POST /embed/batch` — list of texts, batch_size=128 internally
- `GET /health`
- Port 8021

**Benchmarks** (nomic-embed-text-v2-moe on Vega 48 MPS):

| Mode | Throughput |
|------|-----------|
| Sequential single | 3.4 embeds/sec (~293ms/req) |
| Batch 128 | 28.3 embeds/sec (35ms/embed) |
| Batch 256 | 25.4 embeds/sec (VRAM pressure) |
| 4 workers × batch 128 | 102.5 embeds/sec |
| 8 workers × batch 128 | 111.8 embeds/sec (peak) |

Batch size 128 is the sweet spot. 256 starts dropping off from VRAM pressure on the 8GB card.

### Vision Captioning — `moondream_service.py`

[Moondream2](https://huggingface.co/vikhyatk/moondream2) (1.8B) running in fp16 on MPS. Captioning, visual Q&A, and batch captioning.

- `POST /caption` — base64 image → caption (supports short/normal/long)
- `POST /query` — base64 image + question → answer
- `POST /caption/batch` — list of base64 images → captions (sequential on GPU)
- `GET /health`
- Port 8022
- ~22s/image, ~3.5GB VRAM

### Image Generation — `sdxl_service.py`

[SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) in fp16 on MPS with DPMSolverMultistepScheduler, attention slicing, and VAE slicing for the 8GB VRAM limit.

- `POST /generate` — prompt → base64 PNG (configurable steps, size, guidance, negative prompt)
- `GET /health`
- Port 8023
- ~8min/image at 1024×1024, 35 steps
- ~6.5GB VRAM — stop other services for best results

Uses DPMSolverMultistepScheduler because the default Euler scheduler has an off-by-one IndexError in diffusers 0.30.x.

### Queue Runner — `sdxl_queue.py`

Standalone script that pulls prompts from the Mac Studio over SSH, generates images via the SDXL service, and saves PNGs locally. Resumable — logs progress to `queue_log.json` and skips completed items on restart. Truncates prompts to 60 words to stay within CLIP's 77-token limit.

## VRAM

The Vega 48 has 8GB. Nomic + Moondream fit together (~5.5GB). SDXL needs most of it (~6.5GB) — stop other services first.

## Setup

```bash
pip install -r requirements.txt

# Embeddings
python nomic_embed_service.py

# Vision captioning
uvicorn moondream_service:app --host 0.0.0.0 --port 8022 --workers 1

# Image generation
uvicorn sdxl_service:app --host 0.0.0.0 --port 8023 --workers 1
```

## Notes

- All services bind to the Tailscale IP (100.69.11.31), not localhost
- Moondream's 2025-01-09 revision uses `tuple[int, int]` syntax (Python 3.9+) — add `from __future__ import annotations` to `image_crops.py` in the HF cache if on 3.8
- Should work on any Intel Mac with AMD discrete GPU (Vega 56/64, 5500 XT, 5700 XT, W5700X, etc.)

## License

MIT
