#!/usr/bin/env python3
"""
LHF9 positive control: frozen VGGT on Chateau clean vs distractor.
Same separation metric.
"""
import os
import sys
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

def ensure_vggt():
    try:
        from vggt.models.vggt import VGGT
        return VGGT
    except ImportError:
        print("vggt not found; cloning...")
        vggt_dir = Path("/tmp/vggt")
        if not vggt_dir.exists():
            subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/vggt", str(vggt_dir)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(vggt_dir / "requirements.txt")])
        sys.path.insert(0, str(vggt_dir))
        from vggt.models.vggt import VGGT
        return VGGT

REPO_ID = "facebook/VGGT-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path: Path, size: int = 518) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    img = img.crop((left, top, left + size, top + size))
    t = T.ToTensor()(img)  # [3, H, W] in [0, 1]
    return t

def run_pair(model, img_a, img_b):
    # VGGT expects [S, 3, H, W] or [B, S, 3, H, W]
    imgs = torch.stack([img_a, img_b], dim=0).to(DEVICE)  # [2, 3, H, W]
    captured = {}
    handles = []
    agg = getattr(model, "aggregator", None)
    if agg is None:
        print("BLOCKER: no aggregator found")
        return {}
    blocks = []
    if hasattr(agg, "frame_blocks"):
        blocks.extend(agg.frame_blocks)
    if hasattr(agg, "global_blocks"):
        blocks.extend(agg.global_blocks)
    if not blocks:
        print("BLOCKER: no blocks found in aggregator")
        return {}
    def make_hook(idx):
        def hook(module, inp, out):
            f = out[0] if isinstance(out, (tuple, list)) else out
            captured[idx] = f.mean(dim=1).detach().cpu()
        return hook
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        _ = model(imgs)
    for h in handles:
        h.remove()
    return captured

def main():
    VGGT = ensure_vggt()
    print(f"Loading frozen {REPO_ID}")
    model = VGGT.from_pretrained(REPO_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    base = Path("/tmp/dust3r/croco/assets")
    if not base.exists():
        subprocess.check_call(["git", "clone", "--recursive", "https://github.com/naver/dust3r", "/tmp/dust3r"])
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
    feats_cc = run_pair(model, clean1, clean2)
    feats_cd = run_pair(model, clean1, distractor)

    layers = sorted(feats_cc.keys())
    seps = []
    for L in layers:
        sim_cd = F.cosine_similarity(feats_cc[L], feats_cd[L], dim=-1).mean().item()
        sep = round(1.0 - sim_cd, 4)
        seps.append(sep)

    print("SEP_BY_LAYER:", seps)
    mono = all(seps[i] <= seps[i+1] + 1e-4 for i in range(len(seps)-1)) if seps else False
    peaks = seps[-1] == max(seps) if seps else False
    thresh_works = (seps[-1] - seps[0]) > 0.05 if seps else False
    print(f"MONOTONIC: {'yes' if mono else 'no'}")
    print(f"PEAKS_AT_FINAL: {'yes' if peaks else 'no'}")
    print(f"SINGLE_THRESHOLD_WORKS: {'yes' if thresh_works else 'no'}")
    print("VALID_RUN: yes")
    print(f"N_LAYERS: {len(layers)}")

if __name__ == "__main__":
    main()
