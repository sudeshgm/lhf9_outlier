#!/usr/bin/env python3
"""
Force attention weights out of VGGT by monkey-patching Attention.forward.
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

def force_attn_capture(model):
    captured = {}
    from vggt.layers.attention import Attention
    def new_forward(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        captured[id(self)] = attn.detach().cpu()
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    Attention.forward = new_forward
    return captured

def main():
    base = Path("/tmp/lhf9_views")
    if not (base / "clean").exists():
        print("BLOCKER: views not found")
        return
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

    captured = force_attn_capture(model)

    imgs = torch.stack([load_image(cp[0]), load_image(dp[0])]).to(DEVICE)
    with torch.no_grad():
        _ = model(imgs)

    seps = []
    for key in sorted(captured.keys()):
        attn = captured[key]
        mass = attn[..., :, -1].mean().item()
        seps.append(round(mass, 4))

    print("SCORE_METHOD: mean attention mass on distractor view (forced explicit attn via monkey-patch)")
    print("VGGT_SEP_BY_LAYER:", seps)
    print(f"VGGT_MONOTONIC: {'yes' if all(seps[i]<=seps[i+1]+1e-4 for i in range(len(seps)-1)) else 'no'}")
    print(f"VGGT_PEAKS_AT_FINAL: {'yes' if seps and seps[-1]==max(seps) else 'no'}")
    print(f"EXTRACTION_OK: {'yes' if seps else 'no'}")

if __name__ == "__main__":
    main()
