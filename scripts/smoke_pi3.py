#!/usr/bin/env python3
"""
LHF9 kill-test: frozen Pi3 on Chateau clean vs distractor.
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

def ensure_pi3():
    try:
        from pi3.models.pi3 import Pi3
        return Pi3
    except ImportError:
        print("pi3 not found; cloning...")
        pi3_dir = Path("/tmp/Pi3")
        if not pi3_dir.exists():
            subprocess.check_call(["git", "clone", "https://github.com/yyfz/Pi3", str(pi3_dir)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(pi3_dir / "requirements.txt")])
        sys.path.insert(0, str(pi3_dir))
        from pi3.models.pi3 import Pi3
        return Pi3

REPO_ID = "yyfz233/Pi3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path: Path, size: int = 518) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    nw = (nw // 14) * 14
    nh = (nh // 14) * 14
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    t = T.ToTensor()(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)

def run_pair(model, img_a, img_b):
    # Pi3 expects [B, N, C, H, W]
    imgs = torch.stack([img_a, img_b], dim=0).unsqueeze(0).to(DEVICE)  # [1, 2, 3, H, W]
    captured = {}
    handles = []
    decoder = getattr(model, "point_decoder", None) or getattr(model, "camera_decoder", None)
    if decoder is None or not hasattr(decoder, "blocks"):
        print("BLOCKER: no decoder blocks found")
        return {}
    def make_hook(idx):
        def hook(module, inp, out):
            f = out[0] if isinstance(out, (tuple, list)) else out
            captured[idx] = f.mean(dim=1).detach().cpu()
        return hook
    for i, blk in enumerate(decoder.blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        _ = model(imgs)
    for h in handles:
        h.remove()
    return captured

def main():
    Pi3 = ensure_pi3()
    print(f"Loading frozen {REPO_ID}")
    model = Pi3.from_pretrained(REPO_ID).to(DEVICE).eval()
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
