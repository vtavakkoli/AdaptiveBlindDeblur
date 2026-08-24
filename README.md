# Adaptive Blind Deblur

[![CI](https://github.com/vtavakkoli/debluring/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/debluring/actions/workflows/ci.yml)
[![Browser Lab](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f)](https://vtavakkoli.github.io/debluring/)
![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Research](https://img.shields.io/badge/status-experimental%20research-6f42c1)

A reproducible, CPU-friendly blind image deblurring framework with **adaptive PSF estimation**, two guarded restoration refinements, a **full-quality native-resolution Docker benchmark**, and a standalone browser playground.

This repository is maintained as its **own experimental implementation**. It is not presented as a reproduction or port of an older paper. Files under `dataset/results/` are retained only as **legacy outputs for regression and side-by-side evaluation**; they are never used as restoration targets or algorithm inputs.

## Browser Lab

The repository includes a polished, zero-dependency interactive playground at [`demo/index.html`](demo/index.html). It runs entirely in the browser: users can attach an image, tune a motion PSF, estimate a dominant blur direction, adjust restoration settings, inspect the PSF, compare before/after with a slider, and export the result.

After GitHub Pages is enabled with **Settings → Pages → Source: GitHub Actions**, `.github/workflows/pages.yml` deploys the `demo/` directory on every relevant push to `main`:

**https://vtavakkoli.github.io/debluring/**

The Browser Lab is intentionally an **interactive approximation** for quick experimentation. The Python/Docker pipeline remains the authoritative full-quality implementation.

## Methods

| Method | CLI identifier | Role | Learned weights |
|---|---|---|---|
| Adaptive Blind Baseline | `baseline` | Multi-scale blind PSF estimation + TV/L0 restoration | No |
| Annealed PnP Refinement | `annealed-pnp` | Gaussian annealing + NLM prior + blur consistency + artifact guard | No |
| Dual-Extreme Refinement | `extreme-channel` | Dark/bright local-extrema guidance + blur consistency + artifact guard | No |

Both refinements reuse the PSF independently estimated by the baseline for the same observed image. This isolates the effect of the restoration refinement and avoids mixing different blur estimates into one comparison.

## What was fixed in v0.4

The quality benchmark no longer applies one small generic kernel/configuration to every image. That shortcut can leave strong motion blur unresolved and can make a refinement appear numerically better while visually producing duplicated edges or ringing.

The full benchmark now:

- uses an explicit profile for every source in `dataset/benchmark_profiles.json`;
- uses the intended PSF search support for each difficult blur, including supports larger than 75 pixels;
- runs the blind optimizer with the full iteration schedule instead of preview loop caps;
- keeps every image at its original decoded dimensions;
- rejects/refuses refinement strength that lowers reblur error by creating excessive high-frequency artifacts;
- treats all previous saved outputs and kernels as **evaluation-only legacy data**.

The CLI remains freely configurable and does not depend on dataset profiles.

## Full-quality native-resolution benchmark

Run the complete experiment with:

```bash
docker compose build test
docker compose run --rm test
```

The command validates and processes all **23 images** under `dataset/image/`:

```text
23 native source images
× 3 current methods
= 69 native-resolution restorations
+ 23 independently estimated PSFs
```

### Hard resolution invariant

No benchmark input is resized or cropped for evaluation:

```text
output height   == source height
output width    == source width
output channels == source channels
```

Any mismatch is a hard failure in both the Docker runner and GitHub Actions.

### Explicit benchmark profiles

`dataset/benchmark_profiles.json` records, per source image:

- PSF support size;
- gamma used for blind estimation;
- sparse-extrema and gradient regularization;
- TV/L0 final-restoration strengths;
- ringing-removal weight.

These are **configuration values only**. Legacy result pixels and legacy kernel values are never fed into current methods.

## Generated report

A successful run creates:

```text
results/
├── report.html
├── report.json
├── SUMMARY.md
└── images/
    ├── 01_<image>/
    │   ├── input.<original-extension>
    │   ├── legacy_result.png       # evaluation only, when available
    │   ├── legacy_kernel.png       # evaluation only, when available
    │   ├── baseline.png
    │   ├── annealed_pnp.png
    │   ├── extreme_channel.png
    │   ├── interim.png
    │   └── kernel.png
    └── ...
```

Open `results/report.html` for the complete visual comparison. `report.json` is the machine-readable experiment record and includes exact dimensions, source SHA-256 hashes, per-image profiles, estimated PSF shapes, environment versions, timing, diagnostics, and Git/UTC metadata.

## Legacy comparison

For a source `dataset/image/<stem>.<ext>`, the benchmark may display:

```text
dataset/results/<stem>_result.png
dataset/results/<stem>_kernel.png
```

They are called **legacy output** and **legacy kernel** throughout the current report.

Rules:

- legacy assets are evaluation-only;
- no legacy image is resized to force a comparison;
- no legacy kernel values are passed to blind PSF estimation;
- PSNR/SSIM vs legacy measure similarity to a previous saved result, **not ground-truth restoration quality**;
- kernel correlation is shown only when the estimated and legacy supports are directly comparable.

## Metrics

The report intentionally shows several diagnostics instead of declaring a winner from one number:

- **Reblur RMSE ↓** — physical measurement consistency after reapplying the estimated PSF.
- **RMSE gain** — improvement relative to the adaptive blind baseline.
- **Sobel sharpness** — edge-energy diagnostic; excessive values can indicate ringing.
- **Laplacian MAD** — high-frequency/noise diagnostic.
- **PSNR / SSIM vs legacy** — regression/fidelity metrics only.
- **Runtime** — baseline and refinement stage timing.

The two refinements contain an **artifact-safety guard**. A candidate that only improves reblur consistency by strongly increasing high-frequency noise is blended back toward the stable baseline or rejected.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Supported/tested runtime: **Python 3.13+**.

Quality gates:

```bash
python -m ruff check src tests scripts
python -m pytest -q tests
```

## CLI

Adaptive blind baseline:

```bash
dark-channel-deblur input.png output.png \
  --method baseline \
  --kernel-size 85 \
  --lambda-tv 0.01 \
  --lambda-l0 0.002 \
  --kernel-output kernel.png
```

Annealed refinement:

```bash
dark-channel-deblur input.png output.png \
  --method annealed-pnp \
  --kernel-size 85 \
  --seed 7
```

Dual-extreme refinement:

```bash
dark-channel-deblur input.png output.png \
  --method extreme-channel \
  --kernel-size 85
```

`--fast` exists only for interactive previews and small development checks. **Do not use it for quality comparisons.**

Important controls include `--kernel-size`, `--gamma`, `--iterations`, `--lambda-dark`, `--lambda-grad`, `--lambda-tv`, `--lambda-l0`, and `--ring-weight`.

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
config = DeblurConfig(
    kernel_size=85,
    lambda_tv=0.01,
    lambda_l0=0.002,
)

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
├── src/dark_channel_deblur/       # implementation
├── tests/                         # unit + regression tests
├── scripts/                       # benchmark/report tooling
├── demo/
│   ├── index.html                 # standalone browser playground
│   └── README.md
├── dataset/
│   ├── image/                     # 23 native benchmark sources
│   ├── results/                   # legacy evaluation assets
│   └── benchmark_profiles.json    # explicit quality profiles
├── docs/
│   ├── METHODS.md
│   ├── BENCHMARKING.md
│   └── PSF_QUALITY.md
├── .github/workflows/
│   ├── ci.yml
│   └── pages.yml                  # deploys demo/ to GitHub Pages
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Docker mounts `dataset/` read-only, so benchmark execution cannot rewrite source or legacy assets. The Docker test image also copies `demo/`, allowing the regression suite to verify that the standalone page ships with the tested repository state.

## CI

GitHub Actions has two independent quality gates plus an independent Pages deployment workflow:

1. **Python 3.13 quality gate** — package installation, Ruff, unit/regression tests.
2. **Full-quality native-resolution benchmark** — all 23 images, all three methods, exact-dimension checks, PSF/profile checks, report generation, and artifact upload.
3. **GitHub Pages deployment** — publishes the standalone `demo/` folder after changes land on `main`.

The report artifact is published as `deblurring-native-resolution-report` and retained for 30 days.

## Documentation

- [`docs/METHODS.md`](docs/METHODS.md) — current algorithm design.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — experiment and evaluation contract.
- [`docs/PSF_QUALITY.md`](docs/PSF_QUALITY.md) — PSF plausibility and support-preservation design.
- [`demo/README.md`](demo/README.md) — standalone Browser Lab usage and scope.
- [`dataset/README.md`](dataset/README.md) — dataset/profile layout.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development and PR requirements.
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes.
- [`NOTICE.md`](NOTICE.md) — asset/provenance notice.

## Citation

If you use this repository itself, cite the software metadata in [`CITATION.cff`](CITATION.cff). The current repository does not claim to be an implementation of a specific published paper.

## Asset and licensing note

Some files under `dataset/`, `examples/`, and legacy result folders predate the current implementation. Their presence is for evaluation/regression and does not imply ownership or a new license for those assets. See [`NOTICE.md`](NOTICE.md).
