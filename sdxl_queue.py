#!/usr/bin/env python3
"""
SDXL generation queue — pulls prompts from Mac Studio, generates on Vega 48
Output: ~/Desktop/sdxl_generated/
"""

import requests, base64, time, os, re, subprocess, json
from PIL import Image
import io
from datetime import datetime

SDXL_URL = "http://100.69.11.31:8023/generate"
OUTPUT_DIR = os.path.expanduser("~/Desktop/sdxl_generated")
STEPS = 35
MAC_STUDIO_SSH = "lunaai@100.77.49.39"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_to_mac")
PROMPTS_DIR = "~/Desktop/prompts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pull 40 prompts from Mac Studio — single SSH call
def fetch_prompts():
    # Dump all non-video, non-AI, non-stats files in one SSH call
    raw = subprocess.check_output([
        "ssh", "-i", SSH_KEY, MAC_STUDIO_SSH,
        f"for f in {PROMPTS_DIR}/*.md; do"
        f"  b=$(basename $f);"
        f"  echo \"===FILE:$b===\";"
        f"  cat \"$f\";"
        f" done"
    ]).decode(errors="replace")

    prompts = []
    current_file = "unknown"
    for line in raw.splitlines():
        if line.startswith("===FILE:"):
            fname = line.replace("===FILE:","").replace("===","").strip()
            # Skip video, AI-enriched, stats, template files
            if any(x in fname.lower() for x in ["video", "ai-", "_stats", "template", "enrich", "i2v", "i2i"]):
                current_file = None
            else:
                current_file = fname.replace(".md","")
        elif current_file:
            m = re.match(r'^\d+\.\s+(.{20,})', line)
            if m:
                prompts.append((current_file, m.group(1).strip()))

    print(f"Parsed {len(prompts)} T2I prompts from original files")

    # Pick 40 spread evenly across all sources
    step = max(1, len(prompts) // 40)
    selected = prompts[::step][:40]
    print(f"Selected {len(selected)} prompts\n")
    return selected

def truncate_prompt(prompt, max_words=60):
    """CLIP only handles 77 tokens — trim to ~60 words to stay safe."""
    words = prompt.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return prompt

def generate(prompt, negative="blurry, bad quality, watermark, text, ugly, deformed"):
    prompt = truncate_prompt(prompt)
    r = requests.post(SDXL_URL, json={
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": STEPS,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 7.5
    }, timeout=600)
    r.raise_for_status()
    return r.json()["image"]  # base64

def save_image(b64, filename):
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, format="PNG")
    return path

# Log file — resume from previous run
log_path = os.path.join(OUTPUT_DIR, "queue_log.json")
results = []
completed = set()
if os.path.exists(log_path):
    with open(log_path) as f:
        results = json.load(f)
    completed = {r["i"] for r in results if r["status"] == "ok"}
    print(f"Resuming — {len(completed)} already done\n")

print("=== SDXL Queue Runner ===")
print(f"Steps: {STEPS} | Output: {OUTPUT_DIR}\n")

prompts = fetch_prompts()
total_start = time.time()

for i, (source, prompt) in enumerate(prompts):
    if (i+1) in completed:
        print(f"[{i+1}/{len(prompts)}] SKIP (already done)")
        continue
    print(f"[{i+1}/{len(prompts)}] {source}")
    print(f"  Prompt: {prompt[:80]}...")

    t0 = time.time()
    try:
        b64 = generate(prompt)
        elapsed = time.time() - t0
        ts = datetime.now().strftime("%H%M%S")
        filename = f"{i+1:02d}_{source}_{ts}.png"
        path = save_image(b64, filename)
        eta_remaining = elapsed * (len(prompts) - i - 1)
        print(f"  Done in {elapsed:.0f}s — saved {filename} | ETA remaining: {eta_remaining/60:.0f}min\n")
        results.append({"i": i+1, "source": source, "prompt": prompt, "file": filename, "time": elapsed, "status": "ok"})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED: {e}\n")
        results.append({"i": i+1, "source": source, "prompt": prompt, "file": None, "time": elapsed, "status": str(e)})

    # Save log after each image
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

total = time.time() - total_start
print(f"\n=== Done ===")
print(f"{len([r for r in results if r['status']=='ok'])}/{len(results)} succeeded")
print(f"Total time: {total/60:.1f} min | Avg: {total/len(results):.0f}s/image")
print(f"Output: {OUTPUT_DIR}")
