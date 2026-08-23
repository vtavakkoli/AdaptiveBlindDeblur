# Dark Channel Deblur — fast Python port + research refinements

A modern, tested Python implementation of the blind image deblurring method from:

> Jinshan Pan, Deqing Sun, Hanspeter Pfister, Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

The repository ports the supplied MATLAB algorithm and adds two lightweight research extensions for studying how a classical blind-deblurring prior can be combined with ideas used by more recent restoration methods.

## Three restoration methods

### 1. Dark Channel Baseline (`baseline`)

The baseline follows the released CVPR 2016 flow: coarse-to-fine blind kernel estimation, L0 dark-channel sparsity, L0 gradient sparsity, salient-gradient selection, FFT-based PSF estimation, kernel pruning/centering, and TV/L0 final restoration.

### 2. Annealed Gaussian PnP (`annealed-pnp`)

A **diffusion-inspired, weight-free** plug-and-play refinement. It is deliberately not presented as a trained diffusion model. Starting from the baseline result and its estimated kernel, it iterates:

1. an annealed Gaussian-noise schedule;
2. a fast non-local-means denoising prior;
3. a closed-form FFT data-consistency step for the measured blur model;
4. deterministic candidate selection using reblur consistency plus a small high-frequency/noise diagnostic.

This keeps the useful restoration pattern of modern diffusion/PnP methods—alternate an image prior with explicit measurement consistency—without PyTorch, checkpoints, GPU requirements or neural-network downloads.

### 3. Extreme-Channel Guided (`extreme-channel`)

A second weight-free refinement motivated by **Extreme Channels Prior** work, which combines dark and bright local extrema to handle cases where dark-channel-only assumptions are weak, especially bright or saturated content. This implementation:

1. computes local dark and bright extrema;
2. gates detail recovery more strongly where those extrema are informative;
3. avoids aggressive global contrast stretching;
4. projects every iteration back to the same observed blur model using FFT data consistency.

The two new variants reuse the **same PSF estimated by the baseline**. That isolates the restoration-prior change and prevents the three-method benchmark from spending three times the cost on blind kernel estimation.

## Why the Python implementation is fast

- **OpenCV** morphology, resampling, bilateral filtering and NLM.
- **Numba** JIT compilation for the local-minimum mapping step.
- **SciPy pocketfft** with FFT-friendly shapes and configurable CPU workers.
- **NumPy** vectorized gradients/divergence and thresholding.
- **float32** iterative buffers to reduce memory bandwidth.
- Bulk dark-channel projection instead of repeated overlapping patch copies.
- Shared kernel estimation for both new refinement methods.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python **3.13+** is the supported/tested runtime.

## CLI

Baseline:

```bash
dark-channel-deblur input.png output.png \
  --method baseline \
  --kernel-size 25 \
  --kernel-output kernel.png
```

Annealed Gaussian PnP:

```bash
dark-channel-deblur input.png output_pnp.png \
  --method annealed-pnp \
  --kernel-size 25 \
  --seed 7 \
  --fast
```

Extreme-channel guided refinement:

```bash
dark-channel-deblur input.png output_extreme.png \
  --method extreme-channel \
  --kernel-size 25 \
  --fast
```

Important baseline parameters remain available directly: `--kernel-size`, `--gamma`, `--lambda-dark`, `--lambda-grad`, and `--iterations`.

## Python API

```python
from dark_channel_deblur import (
    DeblurConfig,
    annealed_pnp_refine,
    deblur_image,
    extreme_channel_refine,
)
from dark_channel_deblur.io import read_image, write_image

image = read_image("input.png")
config = DeblurConfig(kernel_size=25)
baseline, kernel, interim = deblur_image(image, config)

pnp = annealed_pnp_refine(image, baseline, kernel, seed=7)
extreme = extreme_channel_refine(image, baseline, kernel)

write_image("baseline.png", baseline)
write_image("annealed_pnp.png", pnp)
write_image("extreme_channel.png", extreme)
```

## Full `dataset/image` Docker benchmark

The repository contains **23 source images** under `dataset/image`. One command now tests every one of them:

```bash
docker compose run --rm test
```

The Docker test performs three phases:

1. verifies that all 23 supported source images are present;
2. runs the unit and regression tests;
3. processes every image using the baseline and both new refinement methods, then creates the final HTML/JSON report.

For CI practicality, each source is resized **only in memory** to a maximum side of 192 pixels. The committed originals are never rewritten. The experiment therefore generates **69 restored outputs** (23 images × 3 methods) while estimating only **23 blind PSFs**, because both refinements share the corresponding baseline kernel.

Generated output structure:

```text
results/
├── report.html
├── report.json
└── images/
    ├── 01_.../
    │   ├── input.png
    │   ├── baseline.png
    │   ├── annealed_pnp.png
    │   ├── extreme_channel.png
    │   ├── interim.png
    │   └── kernel.png
    ├── 02_.../
    └── ... 23 cases total
```

Open **`results/report.html`** after the Docker run. It contains aggregate metrics plus a collapsible visual comparison for every source image.

### Report metrics

Most files in `dataset/image` are real/natural examples without paired clean ground truth, so the report does not invent PSNR/SSIM values for them. Instead it records:

- **Reblur RMSE** — reblur the restored image using the shared estimated PSF and compare it with the observed input. Lower means stronger measurement consistency.
- **Sobel sharpness** — mean gradient magnitude. Higher edge energy can indicate useful recovery but may also reward ringing, so it is a diagnostic rather than a stand-alone quality score.
- **Laplacian MAD** — a high-frequency/noise diagnostic.
- **Dark fraction / bright fraction** — local extreme-channel sparsity indicators.
- **Runtime** for every method.

For the explicit blurred/clean pairs present in the folder—`26.blurred → 26` and `flower_blurred → flower`—the report additionally computes **PSNR and SSIM**.

The 192-pixel CI resolution is for reproducible automated validation; it is not presented as a full-resolution SOTA benchmark.

## MATLAB reference regression

A same-input historical regression case remains under [`examples/real_img2`](examples/real_img2), including the authors' saved MATLAB result/kernel and previous Python full/fast snapshots.

| Output | PSNR vs MATLAB output | SSIM vs MATLAB output | Kernel correlation |
|---|---:|---:|---:|
| Python full | 31.49 dB | 0.9555 | 0.8805 |
| Python `--fast` | 34.80 dB | 0.9756 | 0.9524 |
| Blurred input | 25.03 dB | 0.7797 | — |

Those are **agreement metrics against the authors' MATLAB output, not ground-truth quality metrics**.

## Tests and CI

Run the unit tests directly:

```bash
python -m pytest -q tests
```

Run the complete Docker validation:

```bash
docker compose build test
docker compose run --rm test
```

GitHub Actions uses **Python 3.13**. The Docker job fails if `dataset/image` does not contain exactly 23 supported images, if one of the three methods produces invalid output, if any per-image result/kernel is missing, or if the final report is incomplete. The complete `results/` directory is uploaded as the **`deblurring-full-dataset-report`** Actions artifact.

## Docker usage for your own image

Create a `data/` directory containing `input.png`, then for example:

```bash
docker compose run --rm deblur \
  /data/input.png /data/output.png \
  --method annealed-pnp \
  --kernel-size 25 --fast
```

## Scientific positioning

The baseline is a port of Pan et al. (CVPR 2016). The two additions are **new experimental variants in this repository**:

- **Extreme-Channel Guided** is motivated by Yan et al., *Image Deblurring via Extreme Channels Prior*, CVPR 2017, which supplements dark-channel evidence with the bright channel.
- **Annealed Gaussian PnP** is motivated by the structure of modern plug-and-play/diffusion restoration: progressively apply a denoising prior while repeatedly enforcing the known degradation model. It uses classical NLM rather than a learned diffusion checkpoint, so calling it a SOTA diffusion model would be incorrect.

A future learned backend could replace NLM with a pretrained diffusion/score prior while keeping the same data-consistency interface. Any SOTA claim should then be evaluated on standard paired deblurring benchmarks at their intended resolution, with perceptual and distortion metrics, rather than on this 23-image qualitative release folder alone.

## Notes on baseline fidelity

The baseline optimization objective and coarse-to-fine kernel-estimation sequence follow the supplied CVPR 2016 MATLAB code. Two implementation details are intentionally modernized for speed and robustness:

1. boundary extension uses a vectorized smooth periodic extension rather than the external sine-transform Poisson helper;
2. dark-channel auxiliary updates apply selected local-minimum changes in bulk instead of order-dependent overlapping patch copies.

Exact bit-for-bit MATLAB reproduction is not the goal.

## Citation

If the baseline method is used in academic work, cite the original paper:

```bibtex
@inproceedings{pan2016blind,
  title={Blind Image Deblurring Using Dark Channel Prior},
  author={Pan, Jinshan and Sun, Deqing and Pfister, Hanspeter and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```

For work derived from the bright/dark extreme-channel idea, cite the corresponding Extreme Channels Prior paper as well.
