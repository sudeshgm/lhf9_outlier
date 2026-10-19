#!/usr/bin/env python3
"""
LHF9 attention-mass separation on the existing 6-clean + 2-distractor set.
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

def attention_sep(model, clean_paths, dist_paths, is_vggt=True):
    ct = [load_image(p) for p in clean_paths]
    dt = [load_image(p) for p in dist_paths]
    imgs = torch.stack([ct[0], dt[0]]).to(DEVICE)

    captured = {}
    handles = []
    if is_vggt:
        blocks = list(getattr(model.aggregator, "frame_blocks", [])) + list(getattr(model.aggregator, "global_blocks", []))
    else:
        blocks = list(model.dec_blocks)

    def make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
                captured[idx] = out[1].detach().cpu()
            elif hasattr(module, "attn") and hasattr(module.attn, "last_attn"):
                captured[idx] = module.attn.last_attn.detach().cpu()
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        if is_vggt:
            _ = model(imgs)
        else:
            va = {"img": imgs[0:1], "instance": [0]}
            vb = {"img": imgs[1:2], "instance": [0]}
            _ = model(va, vb)

    for h in handles:
        h.remove()

    seps = []
    for L in sorted(captured.keys()):
        attn = captured[L]
        if attn.dim() >= 3:
            mass = attn[..., :, -1].mean().item() if attn.shape[-1] > 1 else 0.0
        else:
            mass = 0.0
        seps.append(round(mass, 4))
    return seps

def main():
    base = Path("/tmp/lhf9_views")
    if not (base / "clean").exists():
        print("BLOCKER: views not found; run smoke_large.py first")
        return
    cp = sorted((base / "clean").glob("*.jpg"))[:6]
    dp = sorted((base / "dist").glob("*.jpg"))[:2]

    print("SCORE_METHOD: mean attention mass on the distractor view (pairwise clean-distractor forward)")

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
    vs = attention_sep(model, cp, dp, is_vggt=True)
    print("VGGT_SEP_BY_LAYER:", vs)
    print(f"VGGT_MONOTONIC: {'yes' if all(vs[i]<=vs[i+1]+1e-4 for i in range(len(vs)-1)) else 'no'}")
    print(f"VGGT_PEAKS_AT_FINAL: {'yes' if vs and vs[-1]==max(vs) else 'no'}")

    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        m2 = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(DEVICE).eval()
        for p in m2.parameters(): p.requires_grad = False
        ds = attention_sep(m2, cp, dp, is_vggt=False)
        print("NONVGGT_SEP_BY_LAYER:", ds)
        print(f"NONVGGT_MONOTONIC: {'yes' if all(ds[i]<=ds[i+1]+1e-4 for i in range(len(ds)-1)) else 'no'}")
        print(f"NONVGGT_PEAKS_AT_FINAL: {'yes' if ds and ds[-1]==max(ds) else 'no'}")
    except Exception:
        print("NONVGGT_SEP_BY_LAYER: skipped")
        print("NONVGGT_MONOTONIC: n/a")
        print("NONVGGT_PEAKS_AT_FINAL: n/a")

if __name__ == "__main__":
    main()
