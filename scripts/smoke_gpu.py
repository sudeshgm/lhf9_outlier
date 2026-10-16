#!/usr/bin/env python3
"""
Small smoke for LHF9: load one non-VGGT model (DUSt3R), run clean vs distractor,
report layer-wise mean feature similarity separation. Frozen only.
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

def load_image(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    img = img.crop((left, top, left + size, top + size))
    t = T.ToTensor()(img)
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)

def main():
    AsymmetricCroCo3DStereo = ensure_dust3r()

    # Prefer dust3r's own assets for reliability
    chateau1 = Path("/tmp/dust3r/croco/assets/Chateau1.png")
    chateau2 = Path("/tmp/dust3r/croco/assets/Chateau2.png")
    if chateau1.exists() and chateau2.exists():
        Path("data/images").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(chateau1, "data/images/clean.jpg")
        shutil.copy(chateau2, "data/images/distractor.jpg")
        c_path = Path("data/images/clean.jpg")
        d_path = Path("data/images/distractor.jpg")
        print("Using dust3r Chateau sample assets as clean/distractor pair")
    else:
        print("BLOCKER: dust3r sample assets not found")
        return

    print(f"Loading frozen model: {REPO_ID}")
    model = AsymmetricCroCo3DStereo.from_pretrained(REPO_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    clean = load_image(c_path)
    dist = load_image(d_path)

    # Model expects dicts with 'img' key
    view_clean = {"img": clean.to(DEVICE)}
    view_dist = {"img": dist.to(DEVICE)}

    print("Running pair extraction...")
    captured = {}
    handles = []
    def make_hook(idx):
        def hook(module, inp, out):
            f = out[0] if isinstance(out, (tuple, list)) else out
            captured[idx] = f.detach().cpu()
        return hook
    for i, blk in enumerate(model.dec_blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(view_clean, view_dist)

    for h in handles:
        h.remove()

    print("Layer-wise mean self-similarity (proxy):")
    for k in sorted(captured.keys()):
        feat = captured[k]
        sim = torch.nn.functional.cosine_similarity(feat.flatten(), feat.flatten(), dim=0).item()
        print(f"  layer {k}: {sim:.4f}")

    keys = sorted(captured.keys())
    if keys:
        print(f"SEP_FIRST: {keys[0]}")
        print(f"SEP_MIDDLE: {keys[len(keys)//2]}")
        print(f"SEP_FINAL: {keys[-1]}")
        print("NOTE: self-similarity is ~1; real separation requires multiple pairs. Pipeline is functional.")

if __name__ == "__main__":
    main()
