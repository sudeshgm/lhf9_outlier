# LHF9 Side-by-Side Comparison

Same frozen protocol on Chateau1/2 (clean) vs arch.jpg/solid (distractor).

| Model | SEP_BY_LAYER | MONOTONIC | PEAKS_AT_FINAL | SINGLE_THRESHOLD_WORKS | VALID_RUN |
|-------|--------------|-----------|----------------|------------------------|-----------|
| facebook/VGGT-1B | [0.2271, 0.3922, 0.4043, 0.3709, 0.1181, 0.3481, 0.3527, 0.3441, 0.3409, 0.3185, 0.3027, 0.2818, 0.2992, 0.3631, 0.3106, 0.3723, 0.3514, 0.284, 0.2383, 0.2116, 0.3024, 0.2668, 0.2546, 0.3028, 0.1684, 0.2233, 0.2442, 0.2407, 0.253, 0.244, 0.2294, 0.2225, 0.2181, 0.2003, 0.1731, 0.1555, 0.2215, 0.2079, 0.2331, 0.2455, 0.2357, 0.2208, 0.1582, 0.2118, 0.2217, 0.1862, 0.1845, 0.2223] | no | no | no | yes |
| naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt | [0.1269, 0.1631, 0.2036, 0.1472, 0.1256, 0.1055, 0.0965, 0.0906, 0.0926, 0.1101, 0.1076, 0.0347] | no | no | no | yes |
| yyfz233/Pi3 | [0.0179, 0.0189, 0.013, 0.0083, 0.0789] | no | yes | yes | yes |

## Notes
- All three models were run frozen-only on the same Chateau clean/distractor pair.
- The claimed monotonic final-layer clean/distractor separation did not appear on VGGT either under this protocol.
