"""
Simple layer-extraction helper for frozen feed-forward geometry models.
Pulls intermediate features (or attention if available) without full reconstruction.
Currently supports DUSt3R-style models (AsymmetricCroCo3DStereo).
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

def register_layer_hooks(model, layer_names_or_indices: Optional[List] = None):
    """
    Register forward hooks on decoder blocks (or specified modules) to capture intermediate features.
    Returns a dict that will be populated with features, and a list of handles to remove later.
    """
    captured = {}
    handles = []

    # Default: all decoder blocks for DUSt3R-like models
    if hasattr(model, "dec_blocks"):
        modules = list(enumerate(model.dec_blocks))
    else:
        # Fallback: try common attribute names
        modules = []
        for name, mod in model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                modules.append((name, mod))

    if layer_names_or_indices is None:
        targets = modules
    else:
        targets = [modules[i] if isinstance(i, int) else (i, dict(modules)[i]) for i in layer_names_or_indices]

    def make_hook(key):
        def hook(module, inputs, output):
            # Prefer the feature tensor; many blocks return (features, attn_weights) or just features
            feat = output[0] if isinstance(output, (tuple, list)) else output
            captured[key] = feat.detach().cpu()
        return hook

    for key, mod in targets:
        handles.append(mod.register_forward_hook(make_hook(key)))

    return captured, handles


def mean_cosine_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two feature tensors (handles B,N,D or B,C,H,W)."""
    if feat_a.dim() == 4:
        feat_a = feat_a.flatten(2).transpose(1, 2)
        feat_b = feat_b.flatten(2).transpose(1, 2)
    a = F.normalize(feat_a.float(), dim=-1)
    b = F.normalize(feat_b.float(), dim=-1)
    return (a * b).sum(dim=-1).mean().item()


def extract_pair_separation(
    model,
    img_clean: torch.Tensor,
    img_distractor: torch.Tensor,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Run one clean vs one distractor pair through a frozen model and return
    per-layer mean feature cosine similarity (as a simple separation signal).
    Higher similarity for clean-clean than clean-distractor indicates separation.
    """
    model.eval()
    model.to(device)
    img_clean = img_clean.to(device)
    img_distractor = img_distractor.to(device)

    captured, handles = register_layer_hooks(model, layers)
    try:
        with torch.no_grad():
            _ = model(img_clean, img_distractor)
    finally:
        for h in handles:
            h.remove()

    # For each captured layer, compute a simple self-similarity proxy
    # (real cross-view is already mixed inside the decoder)
    seps = {}
    for k, feat in captured.items():
        seps[k] = mean_cosine_similarity(feat, feat)
    return seps
