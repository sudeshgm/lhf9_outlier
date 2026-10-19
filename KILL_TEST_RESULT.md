# LHF9 Kill Test Result

## Protocol
Frozen-only, view-level clean vs distractor separation on public multi-view assets (Chateau1/2.png as clean pair; arch.jpg or solid-color fallback as distractor).  
No training, no reconstruction metrics, no filtering.

## Models and Results

### DUSt3R
- Model: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`
- N_LAYERS: 12
- SEP_BY_LAYER: [0.1269, 0.1631, 0.2036, 0.1472, 0.1256, 0.1055, 0.0965, 0.0906, 0.0926, 0.1101, 0.1076, 0.0347]
- MONOTONIC: no
- PEAKS_AT_FINAL: no
- SINGLE_THRESHOLD_WORKS: no
- VALID_RUN: yes

### π³ (Pi3)
- Model: `yyfz233/Pi3`
- N_LAYERS: 5
- SEP_BY_LAYER: [0.0179, 0.0189, 0.013, 0.0083, 0.0789]
- MONOTONIC: no
- PEAKS_AT_FINAL: yes
- SINGLE_THRESHOLD_WORKS: yes
- VALID_RUN: yes

## Conclusion
The claimed property (monotonic growth of clean/distractor separation with depth, strictly largest at the final layer) did **not** appear on two valid non-VGGT feed-forward geometry models under the locked protocol.  
Negative result stands.
