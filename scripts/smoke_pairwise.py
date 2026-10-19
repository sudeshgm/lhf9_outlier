#!/usr/bin/env python3
"""
Pairwise view-level cosine. Handle batch dim.
"""
import os, sys, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path, size=518):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left, top = max(0, (nw-size)//2), max(0, (nh-size)//2)
    img = img.crop((left, top, left+size, top+size))
    return T.ToTensor()(img)

def main():
    base = Path("/tmp/lhf9_views")
    cp = sorted((base / "clean").glob("*.jpg"))[:6]
    dp = sorted((base / "dist").glob("*.jpg"))[:2]

    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        d = Path("/tmp/vggt")
        if not d.exists():
            subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/vggt", str(d)])
        sys.path.insert(0, str(d))
        from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad = False

    captured = {}
    handles = []
    blocks = list(getattr(model.aggregator, "frame_blocks", [])) + list(getattr(model.aggregator, "global_blocks", []))
    def make_hook(idx):
        def hook(m, i, o):
            f = o[0] if isinstance(o, (tuple, list)) else o
            captured[idx] = f.detach().cpu()
        return hook
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    all_imgs = torch.stack([load_image(p) for p in cp + dp]).to(DEVICE)
    with torch.no_grad():
        _ = model(all_imgs)
    for h in handles: h.remove()

    seps = []
    n_views = 8
    for L in sorted(captured.keys()):
        feat = captured[L]
        if feat.dim() != 3:
            continue
        if feat.shape[0] > 1:
            feat = feat[0:1]
        B, N, C = feat.shape
        usable = (N // n_views) * n_views
        if usable == 0:
            continue
        feat = feat[:, :usable, :].reshape(n_views, usable // n_views, C)
        view_feats = F.normalize(feat.mean(dim=1), dim=-1)
        cc = [(view_feats[i] * view_feats[j]).sum().item() for i in range(6) for j in range(i+1, 6)]
        cd = [(view_feats[i] * view_feats[j]).sum().item() for i in range(6) for j in range(6, 8)]
        cc_mean = sum(cc)/len(cc) if cc else 0
        cd_mean = sum(cd)/len(cd) if cd else 0
        seps.append(round(cc_mean - cd_mean, 4))

    print("SCORE_METHOD: difference of mean pairwise view-level cosine (clean-clean minus clean-distractor); no global mean-pool")
    print("VGGT_SEP_BY_LAYER:", seps)
    print(f"VGGT_MONOTONIC: {'yes' if all(seps[i]<=seps[i+1]+1e-4 for i in range(len(seps)-1)) else 'no'}")
    print(f"VGGT_PEAKS_AT_FINAL: {'yes' if seps and seps[-1]==max(seps) else 'no'}")

if __name__ == "__main__":
    main()
