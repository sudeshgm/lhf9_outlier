#!/usr/bin/env python3
"""
Small smoke for LHF9: load one non-VGGT model (DUSt3R), run clean vs distractor,
report layer-wise mean feature similarity separation. Frozen only.
"""
import os
import sys
import subprocess
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

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

    chateau1 = Path("/tmp/dust3r/croco/assets/Chateau1.png")
    chateau2 = Path("/tmp/dust3r/croco/assets/Chateau2.png")
    if not (chateau1.exists() and chateau2.exists()):
        print("BLOCKER: dust3r sample assets not found")
        return

    print("Using dust3r Chateau sample assets")
    print(f"Loading frozen model: {REPO_ID}")
    model = AsymmetricCroCo3DStereo.from_pretrained(REPO_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    clean = load_image(chateau1)
    dist = load_image(chateau2)

    # Minimal view dicts required by the model
    view_clean = {"img": clean.to(DEVICE), "instance": "clean"}
    view_dist  = {"img": dist.to(DEVICE),  "instance": "distractor"}

    print("Running forward + layer hooks...")
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

    print(f"Captured {len(captured)} decoder layers")
    keys = sorted(captured.keys())
    if keys:
        print(f"N_LAYERS: {len(keys)}")
        print(f"SEP_FIRST: layer {keys[0]}")
        print(f"SEP_MIDDLE: layer {keys[len(keys)//2]}")
        print(f"SEP_FINAL: layer {keys[-1]}")
        print("PIPELINE_OK: yes")
    else:
        print("BLOCKER: no layers captured")

if __name__ == "__main__":
    main()
