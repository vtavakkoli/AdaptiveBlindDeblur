# Benchmarking protocol

The repository's quality benchmark is:

```bash
docker compose run --rm test
```

It evaluates all 23 supported files under `dataset/image/` with the Adaptive Blind Baseline, Annealed PnP Refinement, and Dual-Extreme Refinement.

## Quality profile

The Docker benchmark is intentionally different from CLI `--fast` preview mode. It runs the full blind optimization and reads per-image configuration from:

```text
dataset/benchmark_profiles.json
```

The profile defines PSF support and regularization parameters for each source. The file must contain exactly one profile for every benchmark image.

No previous result image or previous kernel pixel values are used by inference.

## Native-resolution invariant

Every method receives the decoded source at its original dimensions. The benchmark does not resize or crop a source for evaluation.

For every case:

```text
baseline.shape        == source.shape
annealed_pnp.shape    == source.shape
extreme_channel.shape == source.shape
```

A mismatch is a hard failure. Internal FFT boundary extension is allowed only when the returned restoration is cropped back to the exact source dimensions.

## PSF-support invariant

For every case:

```text
estimated_kernel.shape == (profile.kernel_size, profile.kernel_size)
```

This is also a hard test failure. It prevents a benchmark from silently falling back to a small generic kernel for a long-motion source.

## Legacy evaluation assets

For `dataset/image/<stem>.<ext>`, the report may load:

```text
dataset/results/<stem>_result.png
dataset/results/<stem>_kernel.png
```

These files are **evaluation-only legacy snapshots**.

Rules:

- current methods run before legacy metrics are calculated;
- legacy result pixels are never supplied to any restoration method;
- legacy kernel pixel values are never supplied to blind PSF estimation;
- a legacy result is compared only when its decoded dimensions exactly match the source;
- references are never resized to manufacture comparability;
- kernel correlation is calculated only for directly comparable support sizes.

PSNR/SSIM against a legacy output measure regression/fidelity to a previous saved result, not clean-image ground-truth quality.

## Shared-PSF comparison

Each source receives one blind PSF estimate from the Adaptive Blind Baseline. Both refinements reuse that PSF. This isolates changes caused by the restoration prior rather than mixing them with independent kernel-estimation differences.

## Artifact guard

Reblur RMSE is a physical-consistency diagnostic, but optimizing it alone can reward ringing or duplicated high-frequency structure.

Both refinements therefore compare their high-frequency/noise diagnostic against the stable baseline. A candidate with excessive Laplacian-MAD growth is conservatively blended toward the baseline; a candidate that does not improve reblur consistency is rejected.

## Metrics

The report records:

- **Reblur RMSE** — `RMSE(reblur(restored, estimated_psf), observed)`.
- **RMSE gain** — relative change from the baseline.
- **Sobel sharpness** — edge-energy diagnostic.
- **Laplacian MAD** — high-frequency/noise diagnostic.
- **Runtime** — baseline stage, refinement stage, and aggregate time.
- **PSNR/SSIM vs legacy** — exact-shape regression metrics only.
- **Kernel correlation/L1** — only when current and legacy kernel supports match.

No single diagnostic is treated as a perceptual-quality oracle.

## Reproducibility metadata

`results/report.json` records:

- schema version;
- Git commit and UTC generation time;
- Python/NumPy/SciPy/OpenCV versions;
- source SHA-256;
- native source/output shapes;
- complete per-image benchmark profile;
- current PSF shape, sum, and peak;
- legacy-reference status;
- timing and per-method metrics.

## Output contract

A complete run creates:

```text
results/report.html
results/report.json
results/SUMMARY.md
```

and one directory per source containing the exact source copy, three current restorations, interim latent image, current kernel, and compatible legacy assets when available.

GitHub Actions verifies this contract after Docker completes.

## Developer smoke test

For development only:

```bash
python scripts/generate_report.py --limit 1
```

The Docker/CI quality path never uses `--limit` and expects all 23 sources.

## Reproduce locally

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts
python -m pytest -q tests
docker compose build test
docker compose run --rm test
```

Open `results/report.html` after completion.
