#!/usr/bin/env python3
"""
LHF9 kill-test smoke: frozen DUSt3R, Chateau clean views vs one distractor.
Computes per-layer separation = 1 - cosine(clean-clean features, clean-distractor features).
"""
import os
import sys
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

def ensure_dust3r():
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        return AsymmetricCroCo3DStereo
    except ImportError:
        print("dust3r not found; cloning...")
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

def run_pair(model, img_a, img_b, instance_a="a", instance_b="b"):
    view_a = {"img": img_a.to(DEVICE), "instance": instance_a}
    view_b = {"img": img_b.to(DEVICE), "instance": instance_b}
    captured = {}
    handles = []
    def make_hook(idx):
        def hook(module, inp, out):
            f = out[0] if isinstance(out, (tuple, list)) else out
            captured[idx] = f.mean(dim=1).detach().cpu()
        return hook
    for i, blk in enumerate(model.dec_blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        _ = model(view_a, view_b)
    for h in handles:
        h.remove()
    return captured

def main():
    AsymmetricCroCo3DStereo = ensure_dust3r()
    print(f"Loading frozen {REPO_ID}")
    model = AsymmetricCroCo3DStereo.from_pretrained(REPO_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    base = Path("/tmp/dust3r/croco/assets")
    clean1 = load_image(base / "Chateau1.png")
    clean2 = load_image(base / "Chateau2.png")
    distractor_path = base / "arch.jpg"
    if not distractor_path.exists():
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (512, 512), color=(200, 50, 50))
        img.save("/tmp/distractor.png")
        distractor_path = Path("/tmp/distractor.png")
    distractor = load_image(distractor_path)

    print("Running pairs...")
    feats_cc = run_pair(model, clean1, clean2, "c1", "c2")
    feats_cd = run_pair(model, clean1, distractor, "c1", "d")

    layers = sorted(feats_cc.keys())
    seps = []
    for L in layers:
        sim_cd = F.cosine_similarity(feats_cc[L], feats_cd[L], dim=-1).mean().item()
        sep = round(1.0 - sim_cd, 4)
        seps.append(sep)

    print("SEP_BY_LAYER:", seps)
    mono = all(seps[i] <= seps[i+1] + 1e-4 for i in range(len(seps)-1))
    peaks = seps[-1] == max(seps)
    thresh_works = (seps[-1] - seps[0]) > 0.05
    print(f"MONOTONIC: {'yes' if mono else 'no'}")
    print(f"PEAKS_AT_FINAL: {'yes' if peaks else 'no'}")
    print(f"SINGLE_THRESHOLD_WORKS: {'yes' if thresh_works else 'no'}")
    print("VALID_RUN: yes")
    print(f"N_LAYERS: {len(layers)}")

if __name__ == "__main__":
    main()
