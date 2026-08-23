# Dark Channel Deblur

[![CI](https://github.com/vtavakkoli/debluring/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/debluring/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Research](https://img.shields.io/badge/status-research%20prototype-6f42c1)

A reproducible Python implementation of classical blind image deblurring with a dark-channel prior, extended with two lightweight research refinements and a **native-resolution, full-dataset Docker benchmark**.

The baseline follows:

> Jinshan Pan, Deqing Sun, Hanspeter Pfister, Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

The repository is designed for research comparison rather than as a one-off demo: the package, tests, Docker workflow, historical MATLAB references, metrics, per-image outputs, environment metadata, and HTML report are all versioned or generated reproducibly.

## Methods

| Method | Identifier | Role | Learned weights |
|---|---|---|---|
| Dark Channel Baseline | `baseline` | Blind PSF estimation + TV/L0 restoration | No |
| Annealed Gaussian PnP | `annealed-pnp` | Diffusion-inspired Gaussian annealing + NLM prior + data consistency | No |
| Extreme-Channel Guided | `extreme-channel` | Dark/bright local-extrema refinement + data consistency | No |

Both new methods reuse the **same blind PSF estimated by the baseline for that image**. This isolates the restoration-prior effect and avoids estimating three unrelated kernels.

The Annealed Gaussian PnP method is intentionally described as **diffusion-inspired**, not as a trained diffusion model or a SOTA claim. See [`docs/METHODS.md`](docs/METHODS.md) for the precise algorithmic positioning.

## Native-resolution benchmark

The principal reproducibility command is:

```bash
docker compose build test
docker compose run --rm test
```

The benchmark processes **all 23 images committed under `dataset/image/` at their original decoded dimensions**.

### Resolution contract

The benchmark has a strict invariant:

> **No input resizing, no cropping, and no reference resampling.**

For every image and every method:

```text
output height  == source height
output width   == source width
output channels == source channels
```

Any resolution change is a hard test failure in both the Docker test runner and GitHub Actions.

This is important because resizing the input can materially change blur-kernel estimation, dark-channel statistics, edge structure, and the apparent quality of the restoration. Native resolution makes the visual and numerical comparison meaningful.

## Historical MATLAB comparison

The repository already contains historical result files under `dataset/results/`. During the benchmark, a reference is used only when a same-input result exists at:

```text
dataset/results/<source-stem>_result.png
```

and its dimensions match the source **exactly**.

No historical result is resized to make it fit.

When an exact-dimension historical reference is available, the report computes:

- PSNR vs historical MATLAB/release output;
- SSIM vs historical MATLAB/release output;
- kernel correlation and L1 distance when the historical kernel dimensions also match.

These are **agreement/fidelity metrics against the released historical result, not clean-image ground-truth metrics**.

If a historical kernel is available and has a valid odd square size, its dimensions are used to select the Python benchmark kernel size for that image. The historical **kernel values are never fed into the Python methods**.

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the full protocol.

## Generated report

A successful Docker benchmark creates:

```text
results/
├── report.html
├── report.json
├── SUMMARY.md
└── images/
    ├── 01_<image>/
    │   ├── input.<original-extension>
    │   ├── historical_matlab.png        # when exact-shape reference exists
    │   ├── historical_matlab_kernel.png # when exact kernel comparison exists
    │   ├── baseline.png
    │   ├── annealed_pnp.png
    │   ├── extreme_channel.png
    │   ├── interim.png
    │   └── kernel.png
    └── ...
```

Open `results/report.html` for the visual benchmark. It includes:

- all native-resolution source images;
- historical MATLAB outputs when comparable;
- all three Python outputs;
- Python and historical kernels when comparable;
- per-image timing and diagnostics;
- aggregate metrics;
- exact input/output dimensions;
- SHA-256 hashes of benchmark sources;
- Python, NumPy, SciPy and OpenCV versions;
- Git commit and report generation timestamp.

`results/report.json` is the machine-readable experiment record, and `results/SUMMARY.md` is suitable for CI summaries or experiment logs.

## Metrics

For every source image, the report records:

- **Reblur RMSE** — reblur the restored output with the shared estimated PSF and compare it with the native observed image. Lower means stronger measurement consistency.
- **RMSE improvement vs baseline** — relative physical-consistency change introduced by each refinement.
- **Sobel sharpness** — mean edge energy. This can reflect recovered detail but can also reward ringing.
- **Laplacian MAD** — high-frequency/noise diagnostic used to expose oversharpening trade-offs.
- **Dark/bright-channel fractions** — local extreme-channel diagnostics.
- **Runtime** — baseline stage, refinement stage, and aggregate end-to-end time.
- **PSNR/SSIM vs historical MATLAB** — only for exact-dimension same-input historical results.

No unrelated image is treated as ground truth, and no metric is manufactured by resizing a reference.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Supported/tested runtime: **Python 3.13+**.

Run the quality gates locally:

```bash
python -m ruff check src tests scripts
python -m pytest -q tests
```

## CLI

Baseline:

```bash
dark-channel-deblur input.png output.png \
  --method baseline \
  --kernel-size 25 \
  --kernel-output kernel.png \
  --interim-output interim.png
```

Annealed Gaussian PnP:

```bash
dark-channel-deblur input.png output.png \
  --method annealed-pnp \
  --kernel-size 25 \
  --seed 7
```

Extreme-Channel Guided:

```bash
dark-channel-deblur input.png output.png \
  --method extreme-channel \
  --kernel-size 25
```

For a faster interactive preview, append `--fast`. The reproducible Docker benchmark has its own recorded benchmark profile and does **not** resize inputs.

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
annealed = annealed_pnp_refine(image, baseline, kernel, seed=7)
extreme = extreme_channel_refine(image, baseline, kernel)

write_image("baseline.png", baseline)
write_image("annealed.png", annealed)
write_image("extreme.png", extreme)
```

## Repository structure

```text
.
├── src/dark_channel_deblur/   # package implementation
├── tests/                     # unit + regression tests
├── scripts/                   # benchmark and report tooling
├── dataset/
│   ├── image/                 # 23 native benchmark sources
│   └── results/               # historical MATLAB/release outputs
├── examples/real_img2/        # compact historical regression example
├── docs/                      # methods and benchmarking protocol
├── .github/workflows/ci.yml   # Python + native-resolution Docker CI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

The Docker test mounts `dataset/` **read-only**. Dataset binaries are not baked into the container image and cannot be modified by the benchmark.

## Reproducibility and CI

GitHub Actions contains two independent gates:

1. **Python 3.13 quality gate** — install, Ruff lint, unit/regression tests.
2. **Native-resolution benchmark** — full 23-image × 3-method run, report-contract verification, dimension verification, CI summary, and artifact upload.

The benchmark job is allowed a longer execution window because full-resolution blind deblurring is intentionally more expensive than the previous thumbnail-sized CI run.

The artifact is published as:

```text
deblurring-native-resolution-report
```

and retained for 30 days.

## Baseline fidelity

The Python baseline follows the released CVPR 2016 optimization structure: coarse-to-fine blind kernel estimation, L0 dark-channel sparsity, L0 gradient sparsity, salient-gradient selection, FFT PSF estimation, kernel pruning/centering, and TV/L0 final restoration.

Two implementation details are intentionally modernized:

1. boundary extension uses a vectorized smooth periodic extension rather than the external sine-transform Poisson helper;
2. dark-channel auxiliary updates apply selected local-minimum changes in bulk instead of order-dependent overlapping patch copies.

Exact bit-for-bit MATLAB reproduction is not the goal. Historical-output agreement is reported explicitly where a valid same-size reference exists.

## Research positioning

- **Pan et al., CVPR 2016** provides the dark-channel blind-deblurring baseline.
- **Extreme-Channel Guided** is motivated by the idea of combining dark and bright extrema, as used by Extreme Channels Prior work.
- **Annealed Gaussian PnP** explores the inverse-problem pattern of alternating a denoising prior with explicit measurement consistency. Its current denoiser is classical NLM, so it is lightweight, deterministic and weight-free.

A future learned variant can replace NLM with a trained diffusion/score prior while retaining the same physical data-consistency interface. Any SOTA claim should be evaluated separately on standard paired deblurring benchmarks with established metrics and protocol.

## Documentation

- [`docs/METHODS.md`](docs/METHODS.md) — algorithms and design rationale.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — native-resolution evaluation contract and metric interpretation.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow and PR requirements.
- [`CHANGELOG.md`](CHANGELOG.md) — notable repository changes.
- [`NOTICE.md`](NOTICE.md) — provenance and licensing notice.

## Citation

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

For the original dark-channel method, cite:

```bibtex
@inproceedings{pan2016blind,
  title={Blind Image Deblurring Using Dark Channel Prior},
  author={Pan, Jinshan and Sun, Deqing and Pfister, Hanspeter and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```

## Provenance and licensing

This repository includes a Python reimplementation plus dataset/reference assets derived from or supplied with prior research code. See [`NOTICE.md`](NOTICE.md) before redistributing those assets. No new license is asserted over third-party research assets by their inclusion here.
