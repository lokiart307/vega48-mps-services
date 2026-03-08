# MPS ML Services on Intel Mac + AMD GPU

Three production ML services running on an Intel iMac (2019) with a **Radeon Pro Vega 48 (8GB)** via PyTorch MPS.

**They said it couldn't be done.** MPS is "Apple Silicon only." The Vega 48 is "not supported for ML." This repo proves otherwise.

## What's Running

| Service | Model | Port | Speed | VRAM |
|---------|-------|------|-------|------|
| **Text Embeddings** | nomic-embed-text-v2-moe | 8021 | instant | ~2GB |
| **Vision Captioning** | Moondream2 (1.8B) | 8022 | ~22s/image | ~3.5GB |
| **Image Generation** | SDXL Base 1.0 | 8023 | ~8min/image | ~6.5GB |

## Hardware

- **iMac 2019** (Intel Core i9, 80GB RAM)
- **Radeon Pro Vega 48** (8GB VRAM) — discrete AMD GPU
- macOS 14.6.1 (Sonoma)
- PyTorch 2.2.2, `device="mps"`

## The Point

Every doc, forum post, and AI assistant says MPS requires Apple Silicon. Metal works on Intel Macs with AMD GPUs. PyTorch MPS works on them too. We watched the GPU History in Activity Monitor spike during inference. This is not CPU fallback.

A 2019 iMac sitting idle overnight can:
- Caption 200 images for free
- Generate 40 SDXL images while you sleep
- Embed thousands of text chunks for vector search

No cloud API. No subscription. No new hardware. Just hardware you already own, doing work they told you it couldn't do.

## Quick Start

```bash
pip install -r requirements.txt

# Text embeddings (smallest, run alongside others)
python nomic_embed_service.py

# Vision captioning
uvicorn moondream_service:app --host 0.0.0.0 --port 8022 --workers 1

# Image generation (needs most VRAM — stop other services first for best results)
uvicorn sdxl_service:app --host 0.0.0.0 --port 8023 --workers 1
```

## API Usage

### Embeddings
```bash
curl -X POST http://localhost:8021/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
# → {"embedding": [0.012, -0.034, ...]}  (768 dims)

# Batch
curl -X POST http://localhost:8021/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'
```

### Vision Captioning
```python
import base64, requests

with open("photo.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

# Caption
r = requests.post("http://localhost:8022/caption", json={"image": img, "length": "normal"})
print(r.json()["caption"])

# Visual Q&A
r = requests.post("http://localhost:8022/query", json={"image": img, "question": "What's in this image?"})
print(r.json()["answer"])
```

### Image Generation
```python
import base64, requests
from PIL import Image
import io

r = requests.post("http://localhost:8023/generate", json={
    "prompt": "a lighthouse on rocky cliffs at sunset, dramatic clouds, golden hour",
    "steps": 35,
    "width": 1024,
    "height": 1024
})
img = Image.open(io.BytesIO(base64.b64decode(r.json()["image"])))
img.save("output.png")
```

## Notes

- **VRAM management**: The Vega 48 has 8GB. Nomic + Moondream fit together (~5.5GB). SDXL needs most of it (~6.5GB) — stop other services for best results.
- **Moondream Python 3.8 fix**: The 2025-01-09 revision uses `tuple[int, int]` syntax (Python 3.9+). Add `from __future__ import annotations` to the top of `~/.cache/huggingface/modules/transformers_modules/vikhyatk/moondream2/.../image_crops.py` if running Python 3.8.
- **SDXL scheduler**: Uses DPMSolverMultistepScheduler — the default Euler scheduler has an off-by-one IndexError in diffusers 0.30.x.
- **Batch captioning**: Moondream's `/caption/batch` endpoint processes lists of images sequentially on GPU — no HTTP overhead between images. ~22s/image average.

## Tested On

- iMac 2019, Radeon Pro Vega 48 (8GB), macOS Sonoma 14.6.1
- Should work on any Intel Mac with AMD discrete GPU (Vega 56/64, 5500 XT, 5700 XT, W5700X, etc.)

## License

MIT — do whatever you want with it.
