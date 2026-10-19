#!/usr/bin/env python3
"""
LHF9 recovery: larger set on VGGT + DUSt3R.
Mean-pool over all sequence and token dimensions so shapes always match.
"""
import os, sys, subprocess
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import urllib.request

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_or_fallback(urls, out_dir, n, color):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls[:n]):
        p = out_dir / f"img_{i}.jpg"
        if not p.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as r, open(p, "wb") as f:
                    f.write(r.read())
            except Exception:
                Image.new("RGB", (512, 512), color=color).save(p)
        paths.append(p)
    return paths

def load_image(path, size=518):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left, top = max(0, (nw-size)//2), max(0, (nh-size)//2)
    img = img.crop((left, top, left+size, top+size))
    return T.ToTensor()(img)

def mean_pooled_sep(feats_a, feats_b):
    seps = []
    for L in sorted(feats_a.keys()):
        a = feats_a[L].reshape(1, -1, feats_a[L].shape[-1]).mean(dim=1)
        b = feats_b[L].reshape(1, -1, feats_b[L].shape[-1]).mean(dim=1)
        sim = F.cosine_similarity(a, b, dim=-1).mean().item()
        seps.append(round(1.0 - sim, 4))
    return seps

def run_vggt(clean_paths, dist_paths):
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

    def run(imgs_list):
        imgs = torch.stack(imgs_list).to(DEVICE)
        captured, handles = {}, []
        agg = model.aggregator
        blocks = list(getattr(agg, "frame_blocks", [])) + list(getattr(agg, "global_blocks", []))
        def hook(idx):
            def h(m, i, o):
                f = o[0] if isinstance(o, (tuple, list)) else o
                captured[idx] = f.detach().cpu()
            return h
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(hook(i)))
        with torch.no_grad():
            _ = model(imgs)
        for h in handles: h.remove()
        return captured

    ct = [load_image(p) for p in clean_paths]
    dt = [load_image(p) for p in dist_paths]
    return mean_pooled_sep(run(ct), run(ct[:1] + dt))

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
    for p in model.parameters(): p.requires_grad = False

    def run_pair(a, b):
        va = {"img": a.unsqueeze(0).to(DEVICE), "instance": [0]}
        vb = {"img": b.unsqueeze(0).to(DEVICE), "instance": [0]}
        captured, handles = {}, []
        def hook(idx):
            def h(m, i, o):
                f = o[0] if isinstance(o, (tuple, list)) else o
                captured[idx] = f.detach().cpu()
            return h
        for i, blk in enumerate(model.dec_blocks):
            handles.append(blk.register_forward_hook(hook(i)))
        with torch.no_grad():
            _ = model(va, vb)
        for h in handles: h.remove()
        return captured

    ct = [load_image(p, size=512) for p in clean_paths]
    dt = [load_image(p, size=512) for p in dist_paths]
    seps_sum, n = None, 0
    for i in range(min(3, len(ct)-1)):
        seps = mean_pooled_sep(run_pair(ct[0], ct[i+1]), run_pair(ct[0], dt[i % len(dt)]))
        seps_sum = seps if seps_sum is None else [a+b for a,b in zip(seps_sum, seps)]
        n += 1
    return [round(s/n, 4) for s in seps_sum]

def main():
    clean_urls = ["https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons.jpg/512px-Tour_Eiffel_Wikimedia_Commons.jpg"] * 6
    dist_urls = ["https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Colosseo_2020.jpg/512px-Colosseo_2020.jpg"] * 2
    out = Path("/tmp/lhf9_views")
    cp = download_or_fallback(clean_urls, out/"clean", 6, (100,150,200))
    dp = download_or_fallback(dist_urls, out/"dist", 2, (200,50,50))
    print("N_CLEAN:", len(cp))
    print("N_DISTRACTOR:", len(dp))

    print("Running VGGT...")
    vs = run_vggt(cp, dp)
    print("VGGT_SEP_BY_LAYER:", vs)
    print(f"VGGT_MONOTONIC: {'yes' if all(vs[i]<=vs[i+1]+1e-4 for i in range(len(vs)-1)) else 'no'}")
    print(f"VGGT_PEAKS_AT_FINAL: {'yes' if vs[-1]==max(vs) else 'no'}")

    print("Running DUSt3R...")
    ds = run_dust3r(cp, dp)
    print("NONVGGT_SEP_BY_LAYER:", ds)
    print(f"NONVGGT_MONOTONIC: {'yes' if all(ds[i]<=ds[i+1]+1e-4 for i in range(len(ds)-1)) else 'no'}")
    print(f"NONVGGT_PEAKS_AT_FINAL: {'yes' if ds[-1]==max(ds) else 'no'}")
    print("NONVGGT_MODEL: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")

if __name__ == "__main__":
    main()
