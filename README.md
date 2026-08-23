# Dark Channel Deblur — fast Python port + research refinements

A modern, tested Python implementation of the blind image deblurring method from:

> Jinshan Pan, Deqing Sun, Hanspeter Pfister, Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

The repository ports the supplied MATLAB algorithm and now includes two lightweight research extensions for studying how a classical blind-deblurring prior can be combined with ideas used by more recent restoration methods.

## Methods

### 1. Dark Channel Baseline (`baseline`)

The baseline follows the released CVPR 2016 flow: coarse-to-fine blind kernel estimation, L0 dark-channel sparsity, L0 gradient sparsity, salient-gradient selection, FFT-based PSF estimation, kernel pruning/centering, and TV/L0 final restoration.

### 2. Annealed Gaussian PnP (`annealed-pnp`)

A **diffusion-inspired, weight-free** plug-and-play refinement. It deliberately does **not** claim to be a trained diffusion model. Starting from the baseline result and its estimated kernel, it iterates:

1. an annealed Gaussian-noise schedule;
2. a fast non-local-means denoising prior;
3. a closed-form FFT data-consistency step for the measured blur model;
4. deterministic candidate selection using reblur consistency plus a small noise diagnostic.

This mirrors a useful pattern in modern diffusion/PnP restoration—alternate a denoising/image prior with explicit measurement consistency—without introducing model checkpoints, PyTorch, GPU requirements, or network inference.

### 3. Extreme-Channel Guided (`extreme-channel`)

A second weight-free refinement motivated by **Extreme Channels Prior** work, which observes that using both dark and bright local extrema helps with scenes where dark-channel-only assumptions are weak, especially bright/saturated content. The implementation:

1. computes local dark and bright extrema;
2. gates edge/detail recovery more strongly where those extrema are informative;
3. avoids global contrast stretching;
4. projects every iteration back to the same observed blur model with FFT data consistency.

The two research variants reuse the **same PSF estimated by the baseline**. This isolates the effect of the restoration prior and prevents a three-method report from spending 3× the cost on blind kernel estimation.

## Why this Python port is fast

- **OpenCV** morphology, resampling, bilateral filtering and NLM.
- **Numba** JIT compilation for the local-minimum mapping step.
- **SciPy pocketfft** with FFT-friendly shapes and configurable CPU workers.
- **NumPy** vectorized gradients/divergence and thresholding.
- **float32** iterative buffers to reduce memory bandwidth.
- Bulk dark-channel projection instead of repeated overlapping patch copies.
- Shared kernel estimation for the two new refinements.

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

Extreme-channel refinement:

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

## Full 23-image Docker benchmark

The authors' official project page links the original `cvpr16_deblurring_code_v1.zip`. The Docker test downloads that package, extracts **all 23 supported images** from its `image/` directory into `dataset/image/`, and creates bounded 192 px lossless CI working copies. The original stems are retained.

Run the complete experiment with one command:

```bash
docker compose run --rm test
```

The command performs three phases:

1. prepares all 23 official source images;
2. runs unit/regression tests;
3. processes every image with the baseline and both refinements and builds the final report.

That produces **69 restored images** (23 × 3 methods), while doing only 23 blind PSF estimations because the two refinements share each image's baseline kernel.

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

Open **`results/report.html`** after the run. It contains aggregate tables plus collapsible visual comparisons for every image.

### Report metrics

Most files in the release are real/natural examples without paired clean ground truth, so the report does **not** invent PSNR/SSIM for them. Instead it records:

- **Reblur RMSE** — reblur the restored image with the shared estimated kernel and compare it to the observed input. Lower means better measurement consistency.
- **Sobel sharpness** — mean gradient magnitude. Higher edge energy is useful but can also indicate ringing, so it is diagnostic rather than a stand-alone quality score.
- **Laplacian MAD** — a noise/high-frequency diagnostic.
- **Dark fraction / bright fraction** — local extreme-channel sparsity indicators.
- **Runtime** for each method.

For the two explicit blurred/clean pairs in the supplied image folder—`26.blurred → 26` and `flower_blurred → flower`—the report additionally computes **PSNR and SSIM**.

The 192 px scaling is a CI working resolution, not a claim to reproduce full-resolution benchmark numbers from the papers.

## MATLAB reference regression

A same-input historical regression case remains under [`examples/real_img2`](examples/real_img2), including the authors' saved MATLAB result/kernel and previous Python full/fast snapshots.

| Output | PSNR vs MATLAB output | SSIM vs MATLAB output | Kernel correlation |
|---|---:|---:|---:|
| Python full | 31.49 dB | 0.9555 | 0.8805 |
| Python `--fast` | 34.80 dB | 0.9756 | 0.9524 |
| Blurred input | 25.03 dB | 0.7797 | — |

Those are **agreement metrics against the authors' MATLAB output, not ground-truth quality metrics**.

## Tests and CI

Run unit tests directly:

```bash
python -m pytest -q tests
```

The complete Docker validation is:

```bash
docker compose build test
docker compose run --rm test
```

GitHub Actions uses **Python 3.13**. The Docker job fails if it does not obtain exactly 23 dataset images, if one of the three methods produces invalid output, if any per-image result/kernel is missing, or if the final report is incomplete. The full `results/` directory is uploaded as the `deblurring-full-dataset-report` artifact.

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

- The **Extreme-Channel Guided** variant is motivated by Yan et al., *Image Deblurring via Extreme Channels Prior*, CVPR 2017, which extends dark-channel reasoning with the bright channel.
- The **Annealed Gaussian PnP** variant is motivated by the structure of modern plug-and-play/diffusion restoration: progressively denoise while repeatedly enforcing the known degradation model. It uses classical NLM rather than a learned diffusion checkpoint, so calling it a SOTA diffusion model would be incorrect.

A future learned extension could replace the NLM denoiser with a pretrained diffusion/score prior (for example a DiffPIR/texture-prior style model) while keeping the same data-consistency interface. That would be a separate optional GPU backend and should be benchmarked on standard paired deblurring datasets before any SOTA claim.

## Notes on fidelity

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

For work derived from the bright/dark extreme-channel idea, also cite the corresponding Extreme Channels Prior paper rather than attributing that idea to this implementation.
