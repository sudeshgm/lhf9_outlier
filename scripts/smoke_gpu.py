#!/usr/bin/env python3
"""
Small smoke for LHF9: load one non-VGGT model (DUSt3R), run clean vs distractor,
report layer-wise mean feature similarity separation. Frozen only.
Uses data/manifest.json. Downloads images on the fly if missing.
"""
import json
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

# Ensure local import
sys.path.insert(0, str(Path(__file__).parent))
from layer_extract import extract_pair_separation

def ensure_dust3r():
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        return AsymmetricCroCo3DStereo
    except ImportError:
        print("dust3r not found; cloning and installing...")
        dust3r_dir = Path("/tmp/dust3r")
        if not dust3r_dir.exists():
            subprocess.check_call(["git", "clone", "--recursive", "https://github.com/naver/dust3r", str(dust3r_dir)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(dust3r_dir / "requirements.txt")])
        sys.path.insert(0, str(dust3r_dir))
        from dust3r.model import AsymmetricCroCo3DStereo
        return AsymmetricCroCo3DStereo

REPO_ID = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_if_needed(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    print(f"Downloading {dest.name}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LHF9-Research/0.1 (research script; contact: sudeshgm)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"  failed: {e}")
        return False

def load_image(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    t = T.ToTensor()(img)
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)

def main():
    AsymmetricCroCo3DStereo = ensure_dust3r()

    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        print("BLOCKER: data/manifest.json missing")
        return
    man = json.loads(manifest_path.read_text())
    urls = man.get("image_urls", {})

    # Download a minimal set for smoke (first scene)
    scene = man["scenes"][0]
    needed = scene["clean_views"][:2] + scene["distractor_views"][:2]
    ok = True
    for p in needed:
        url = urls.get(p)
        if url and not download_if_needed(url, Path(p)):
            ok = False

    # Fallback: use dust3r sample assets if downloads failed
    if not ok:
        print("Falling back to dust3r sample assets...")
        chateau1 = Path("/tmp/dust3r/croco/assets/Chateau1.png")
        chateau2 = Path("/tmp/dust3r/croco/assets/Chateau2.png")
        if chateau1.exists() and chateau2.exists():
            Path("data/images").mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(chateau1, "data/images/eiffel_1.jpg")
            shutil.copy(chateau2, "data/images/colosseum_1.jpg")
            ok = True
        else:
            print("BLOCKER: no fallback images available")
            return

    print(f"Loading frozen model: {REPO_ID}")
    model = AsymmetricCroCo3DStereo.from_pretrained(REPO_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    # One clean vs one distractor
    c_path = Path(scene["clean_views"][0])
    d_path = Path(scene["distractor_views"][0])
    if not c_path.exists() or not d_path.exists():
        print("BLOCKER: sample images failed to download")
        return

    clean = load_image(c_path)
    dist = load_image(d_path)

    print("Running pair extraction...")
    seps = extract_pair_separation(model, clean, dist, device=DEVICE)
    print("Layer-wise mean similarity (proxy):")
    for k in sorted(seps.keys()):
        print(f"  layer {k}: {seps[k]:.4f}")

    keys = sorted(seps.keys())
    if keys:
        print(f"SEP_FIRST: {seps[keys[0]]:.4f}")
        print(f"SEP_MIDDLE: {seps[keys[len(keys)//2]]:.4f}")
        print(f"SEP_FINAL: {seps[keys[-1]]:.4f}")

if __name__ == "__main__":
    main()
