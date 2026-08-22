# `real_img2` MATLAB-reference comparison

This directory keeps a reproducible visual regression case from the supplied CVPR 2016 MATLAB release.

## Provenance

- `input_preview.jpg` is a compact high-quality JPEG preview of the release's `image/real_img2.png`.
- `matlab_reference/` contains a compact JPEG preview of the authors' saved final result plus the exact 25x25 kernel PNG.
- `python/` contains compact JPEG previews of controlled full and `--fast` reruns plus their exact 25x25 kernel PNGs.
- `metrics.json` records the numerical comparison from the controlled rerun.

The original demo configures this case with a 25x25 kernel, `lambda_dark=0.004`, `lambda_grad=0.004`, `gamma_correct=1.0`, `lambda_tv=0.003`, `lambda_l0=0.0005`, and `weight_ring=1.0`.

## Important interpretation

The saved MATLAB image is a **reference implementation output, not ground truth**. Therefore PSNR and SSIM here measure reproduction/agreement with the released MATLAB result. A higher sharpness score can also reflect ringing or amplified high-frequency noise, so it should not be interpreted by itself as better perceptual quality.

## Controlled rerun

| Output | Runtime in current runner | PSNR vs MATLAB | SSIM vs MATLAB | Kernel correlation vs MATLAB |
|---|---:|---:|---:|---:|
| Python full | 22.23 s | 31.49 dB | 0.9555 | 0.8805 |
| Python `--fast` | 6.40 s | 34.80 dB | 0.9756 | 0.9524 |

Runtime is hardware/environment dependent. The MATLAB runtime was not re-measured because MATLAB/Octave is not installed in the execution environment; the reference output is the image shipped by the original authors.

Recompute comparison metrics with:

```bash
python scripts/compare_reference.py \
  --blurred examples/real_img2/input_preview.jpg \
  --reference-result examples/real_img2/matlab_reference/result_preview.jpg \
  --candidate-result examples/real_img2/python/full_result_preview.jpg \
  --reference-kernel examples/real_img2/matlab_reference/kernel.png \
  --candidate-kernel examples/real_img2/python/full_kernel.png
```

The table above was computed from the lossless PNG sources before preview JPEG encoding. Running the comparison script on the committed JPEG previews gives very similar but not bit-identical values.
