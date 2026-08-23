# Dark Channel Deblur — fast Python port

A modern, tested Python implementation of the blind image deblurring method from:

> Jinshan Pan, Deqing Sun, Hanspeter Pfister, Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

This repository reimplements the supplied MATLAB research code as a Python package and CLI. The kernel-estimation flow follows the released algorithm: coarse-to-fine optimization, an L0 dark-channel prior, L0 gradient sparsity, salient-gradient selection, FFT-based PSF estimation, kernel pruning/centering, and a TV/L0 non-blind restoration stage.

## Why this port is faster

The original release uses nested MATLAB loops for the dark channel and repeatedly copies overlapping patches. This port uses:

- **OpenCV** morphology for the dark-channel minimum filter and image resampling.
- **Numba** JIT compilation for the remaining local-minimum mapping step.
- **SciPy pocketfft** with FFT-friendly sizes and configurable CPU workers.
- **NumPy** vectorized periodic gradients/divergence and thresholding.
- **float32** image buffers in the iterative image solvers to reduce memory bandwidth.
- Bulk dark-channel projection instead of copying an entire patch for every pixel.

The first call includes Numba compilation overhead; later calls reuse the compiled cache.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python **3.13+** is the supported/tested runtime.

## CLI

```bash
dark-channel-deblur input.png output.png \
  --kernel-size 25 \
  --kernel-output kernel.png \
  --interim-output interim.png
```

For a quicker preview:

```bash
dark-channel-deblur input.png output.png --kernel-size 25 --fast
```

Important parameters from the original MATLAB package are available directly: `--kernel-size`, `--gamma`, `--lambda-dark`, `--lambda-grad`, and `--iterations`.

## Python API

```python
from dark_channel_deblur import DeblurConfig, deblur_image
from dark_channel_deblur.io import read_image, write_image

image = read_image("input.png")
config = DeblurConfig(kernel_size=25, gamma_correct=1.0)
result, kernel, interim = deblur_image(image, config)
write_image("output.png", result)
```

## MATLAB reference comparison

A same-input regression case is included under [`examples/real_img2`](examples/real_img2). It contains the blurred test image preview, the authors' saved MATLAB result and kernel, and the previous Python full/fast results and kernels.

| Output | PSNR vs MATLAB output | SSIM vs MATLAB output | Kernel correlation |
|---|---:|---:|---:|
| Python full | 31.49 dB | 0.9555 | 0.8805 |
| Python `--fast` | 34.80 dB | 0.9756 | 0.9524 |
| Blurred input | 25.03 dB | 0.7797 | — |

These are **agreement metrics against the authors' released MATLAB result, not ground-truth quality metrics**. The historical measurements were computed from the original 360×480 sources; compact preview assets are stored in the repository for fast CI regression checks. See [`examples/real_img2/README.md`](examples/real_img2/README.md) and [`metrics.json`](examples/real_img2/metrics.json).

## Tests and visual report

Run the unit and regression tests directly:

```bash
python -m pytest -q tests
```

For the complete reproducible validation, use Docker Compose:

```bash
docker compose build test
docker compose run --rm test
```

The Docker test now performs the full validation workflow:

1. runs the unit/synthetic/reference regression tests;
2. deblurs the included `real_img2` test image again inside the container;
3. writes the fresh result, interim latent image, and estimated kernel into `results/`;
4. compares the new output with the original MATLAB result and the previous Python full/fast snapshots;
5. computes PSNR, SSIM, kernel correlation, and kernel L1 distance;
6. generates a self-contained visual comparison at **`results/report.html`** plus machine-readable **`results/report.json`**.

Typical generated files are:

```text
results/
├── report.html
├── report.json
├── new_python_result.png
├── new_python_interim.png
├── new_python_kernel.png
├── input.jpg
├── matlab_reference.jpg
├── previous_python_full.jpg
├── previous_python_fast.jpg
└── *_kernel.png
```

GitHub Actions uses **Python 3.13** and also runs this Docker Compose workflow. The generated `results/` directory is uploaded as the `deblurring-comparison-report` CI artifact, so the HTML report and images can be inspected after each run.

## Docker usage

Create a `data/` directory containing `input.png`, then run:

```bash
docker compose run --rm deblur \
  /data/input.png /data/output.png \
  --kernel-size 25 --fast
```

## Notes on fidelity

The optimization objective and coarse-to-fine kernel-estimation sequence follow the supplied CVPR 2016 MATLAB code. Two implementation details are intentionally modernized for speed and robustness:

1. The boundary extension uses a vectorized smooth periodic extension rather than the MATLAB package's external sine-transform Poisson helper.
2. The dark-channel auxiliary update applies selected local-minimum changes in bulk, avoiding order-dependent overlapping patch copies.

Those changes make the implementation practical and deterministic in Python while retaining the core dark-channel-prior method. Exact bit-for-bit MATLAB reproduction is not a goal.

## Citation

If this method is used in academic work, cite the original paper:

```bibtex
@inproceedings{pan2016blind,
  title={Blind Image Deblurring Using Dark Channel Prior},
  author={Pan, Jinshan and Sun, Deqing and Pfister, Hanspeter and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```

The original supplied MATLAB README also asks users to cite that paper when using the code to generate academic results.
