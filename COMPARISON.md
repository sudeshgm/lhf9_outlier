# LHF9 Side-by-Side Comparison

Same frozen protocol on Chateau1/2 (clean) vs arch.jpg/solid (distractor).

| Model | SEP_BY_LAYER | MONOTONIC | PEAKS_AT_FINAL | SINGLE_THRESHOLD_WORKS | VALID_RUN |
|-------|--------------|-----------|----------------|------------------------|-----------|
| facebook/VGGT-1B | — | — | — | — | no (env abort: cuDNN / dependency conflict on T4) |
| naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt | [0.1269, 0.1631, 0.2036, 0.1472, 0.1256, 0.1055, 0.0965, 0.0906, 0.0926, 0.1101, 0.1076, 0.0347] | no | no | no | yes |
| yyfz233/Pi3 | [0.0179, 0.0189, 0.013, 0.0083, 0.0789] | no | yes | yes | yes |

## Notes
- VGGT positive control could not complete: `Aborted` after model download due to missing/incompatible `libcudnn_graph` and package conflicts (torch/numpy/pillow pins from azureml env).
- On the two valid non-VGGT models the monotonic final-layer separation property did not appear.
