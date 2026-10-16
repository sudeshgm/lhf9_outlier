# LHF9 Outlier Rejection Kill Test

CPU-side setup for checking whether the layer-wise clean/distractor separation curve appears in a non-VGGT feed-forward geometry model (DUSt3R).

## Contents
- `data/manifest.json`: 8 multi-view sets from Wikimedia Commons public-domain/CC images (clean = different viewpoints of same landmark; distractors = other landmarks).
- `scripts/layer_extract.py`: simple frozen-model intermediate feature extractor.
- `scripts/smoke_gpu.py`: small GPU smoke that loads DUSt3R, runs one clean vs distractor pair, reports layer-wise similarity.

## GPU command
```bash
cd lhf9_outlier
git pull
export HF_HOME="$PWD/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
python scripts/smoke_gpu.py
```
