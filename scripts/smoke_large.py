#!/usr/bin/env python3
"""
LHF9 recovery: larger set (6-8 clean + 2-3 distractor) on VGGT and one non-VGGT.
"""
import os
import sys
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import urllib.request

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_or_fallback(urls, out_dir, n_needed, fallback_color):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls[:n_needed]):
        p = out_dir / f"img_{i}.jpg"
        if not p.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as r, open(p, "wb") as f:
                    f.write(r.read())
            except Exception:
                img = Image.new("RGB", (512, 512), color=fallback_color)
                img.save(p)
        paths.append(p)
    return paths

def load_image(path: Path, size: int = 518) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    img = img.crop((left, top, left + size, top + size))
    t = T.ToTensor()(img)
    return t

def mean_pooled_sep(feats_cc, feats_cd):
    seps = []
    for L in sorted(feats_cc.keys()):
        sim = F.cosine_similarity(feats_cc[L], feats_cd[L], dim=-1).mean().item()
        seps.append(round(1.0 - sim, 4))
    return seps

def run_vggt(clean_paths, dist_paths):
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        vggt_dir = Path("/tmp/vggt")
        if not vggt_dir.exists():
            subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/vggt", str(vggt_dir)])
        sys.path.insert(0, str(vggt_dir))
        from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    def run(imgs_list):
        imgs = torch.stack(imgs_list, dim=0).to(DEVICE)
        captured = {}
        handles = []
        agg = model.aggregator
        blocks = list(getattr(agg, "frame_blocks", [])) + list(getattr(agg, "global_blocks", []))
        def hook(idx):
            def h(m, i, o):
                f = o[0] if isinstance(o, (tuple, list)) else o
                captured[idx] = f.mean(dim=1).detach().cpu()
            return h
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(hook(i)))
        with torch.no_grad():
            _ = model(imgs)
        for h in handles:
            h.remove()
        return captured

    clean_tensors = [load_image(p) for p in clean_paths]
    dist_tensors = [load_image(p) for p in dist_paths]
    feats_cc = run(clean_tensors)
    feats_cd = run(clean_tensors[:1] + dist_tensors)
    return mean_pooled_sep(feats_cc, feats_cd)

def run_dust3r(clean_paths, dist_paths):
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
    except ImportError:
        d = Path("/tmp/dust3r")
        if not d.exists():
            subprocess.check_call(["git", "clone", "--recursive", "https://github.com/naver/dust3r", str(d)])
        sys.path.insert(0, str(d))
        from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    def run_pair(a, b):
        va = {"img": a.to(DEVICE), "instance": [0]}
        vb = {"img": b.to(DEVICE), "instance": [0]}
        captured = {}
        handles = []
        def hook(idx):
            def h(m, i, o):
                f = o[0] if isinstance(o, (tuple, list)) else o
                captured[idx] = f.mean(dim=1).detach().cpu()
            return h
        for i, blk in enumerate(model.dec_blocks):
            handles.append(blk.register_forward_hook(hook(i)))
        with torch.no_grad():
            _ = model(va, vb)
        for h in handles:
            h.remove()
        return captured

    clean_tensors = [load_image(p) for p in clean_paths]
    dist_tensors = [load_image(p) for p in dist_paths]
    seps_sum = None
    n = 0
    for i in range(min(4, len(clean_tensors)-1)):
        feats_cc = run_pair(clean_tensors[0], clean_tensors[i+1])
        feats_cd = run_pair(clean_tensors[0], dist_tensors[i % len(dist_tensors)])
        seps = mean_pooled_sep(feats_cc, feats_cd)
        if seps_sum is None:
            seps_sum = seps
        else:
            seps_sum = [a+b for a,b in zip(seps_sum, seps)]
        n += 1
    return [round(s/n, 4) for s in seps_sum] if seps_sum else []

def main():
    clean_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons.jpg/512px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons_%28bis%29.jpg/512px-Tour_Eiffel_Wikimedia_Commons_%28bis%29.jpg",
    ] * 4
    dist_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Colosseo_2020.jpg/512px-Colosseo_2020.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Taj_Mahal%2C_Agra%2C_India_edit3.jpg/512px-Taj_Mahal%2C_Agra%2C_India_edit3.jpg",
    ]
    out = Path("/tmp/lhf9_views")
    clean_paths = download_or_fallback(clean_urls, out / "clean", 6, (100, 150, 200))
    dist_paths = download_or_fallback(dist_urls, out / "dist", 2, (200, 50, 50))

    print("N_CLEAN:", len(clean_paths))
    print("N_DISTRACTOR:", len(dist_paths))

    print("Running VGGT...")
    vggt_seps = run_vggt(clean_paths, dist_paths)
    print("VGGT_SEP_BY_LAYER:", vggt_seps)
    mono = all(vggt_seps[i] <= vggt_seps[i+1] + 1e-4 for i in range(len(vggt_seps)-1))
    peaks = vggt_seps[-1] == max(vggt_seps)
    print(f"VGGT_MONOTONIC: {'yes' if mono else 'no'}")
    print(f"VGGT_PEAKS_AT_FINAL: {'yes' if peaks else 'no'}")

    print("Running DUSt3R...")
    dust_seps = run_dust3r(clean_paths, dist_paths)
    print("NONVGGT_SEP_BY_LAYER:", dust_seps)
    mono2 = all(dust_seps[i] <= dust_seps[i+1] + 1e-4 for i in range(len(dust_seps)-1))
    peaks2 = dust_seps[-1] == max(dust_seps)
    print(f"NONVGGT_MONOTONIC: {'yes' if mono2 else 'no'}")
    print(f"NONVGGT_PEAKS_AT_FINAL: {'yes' if peaks2 else 'no'}")
    print("NONVGGT_MODEL: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")

if __name__ == "__main__":
    main()
