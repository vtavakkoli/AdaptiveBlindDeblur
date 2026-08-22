# `real_img2` MATLAB-reference comparison

This directory is a reproducible visual regression case from the supplied CVPR 2016 MATLAB release.

## Files and provenance

- `input_compare.jpg` — compact preview of the authors' `image/real_img2.png` test image.
- `matlab_reference/result_compare.jpg` — compact preview of the authors' saved `results/real_img2_result.png`.
- `matlab_reference/kernel.png` — exact authors' saved 25x25 kernel image.
- `python/full_result_compare.jpg` and `full_kernel.png` — Python full-mode result preview and exact kernel image.
- `python/fast_result_compare.jpg` and `fast_kernel.png` — Python `--fast` result preview and exact kernel image.
- `metrics.json` — measurements computed from the **lossless 360x480 PNG sources** before preview encoding.

The original demo uses a 25x25 kernel, `lambda_dark=0.004`, `lambda_grad=0.004`, `gamma_correct=1.0`, `lambda_tv=0.003`, `lambda_l0=0.0005`, `weight_ring=1.0`, and 5 outer iterations.

## Same-input comparison

| Output | Runtime in current runner | PSNR vs MATLAB | SSIM vs MATLAB | Kernel correlation vs MATLAB |
|---|---:|---:|---:|---:|
| Python full | 22.23 s | 31.49 dB | 0.9555 | 0.8805 |
| Python `--fast` | 6.40 s | 34.80 dB | 0.9756 | 0.9524 |
| Blurred input | — | 25.03 dB | 0.7797 | — |

These PSNR/SSIM values measure **agreement with the released MATLAB result**, not accuracy against ground truth. The shipped MATLAB output is a reference implementation result, not a known sharp target.

The comparison also shows that both Python modes are much closer to the authors' output than the blurred input. In this case `--fast` happens to match the saved MATLAB output more closely than the full mode; this does not imply that fast mode is universally higher quality.

Recompute metrics with `scripts/compare_reference.py` when full-resolution result files are available.
