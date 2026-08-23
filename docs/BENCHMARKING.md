# Benchmarking protocol

The official benchmark is:

```bash
docker compose run --rm test
```

It evaluates all 23 supported files under `dataset/image/` with the baseline, Annealed Gaussian PnP, and Extreme-Channel Guided methods.

## Native-resolution invariant

Every method receives the decoded source at its original dimensions. The benchmark does not resize or crop sources, and it never resizes a historical reference to force a comparison.

For every case:

```text
baseline.shape        == source.shape
annealed_pnp.shape    == source.shape
extreme_channel.shape == source.shape
```

A mismatch is a hard test failure.

Internal FFT boundary extension is allowed only when the final restoration returns to the exact source dimensions.

## Historical MATLAB references

For `dataset/image/<stem>.<ext>`, the benchmark looks for:

```text
dataset/results/<stem>_result.png
dataset/results/<stem>_kernel.png
```

A historical result is used for PSNR/SSIM only if its decoded dimensions exactly match the source. No interpolation is used.

These scores mean agreement with the historical MATLAB/release output for the same observed image. They are not clean-image ground-truth scores.

A valid historical kernel may provide the intended odd square kernel size. Its pixel values are never passed into the Python deblurring methods. When the Python and historical kernels have equal dimensions, the report also records normalized correlation and L1 distance.

## Fairness rule

Each source gets one blind PSF estimated by the dark-channel baseline. Both new refinements reuse that exact PSF. This isolates restoration-prior differences from independent kernel-estimation differences.

## Metrics

The report records:

- **Reblur RMSE**: `RMSE(reblur(restored, estimated_psf), observed)`. Lower means stronger measurement consistency.
- **Sobel sharpness**: mean edge energy; useful as a diagnostic but capable of rewarding ringing.
- **Laplacian MAD**: high-frequency/noise diagnostic.
- **Dark/bright fractions**: local extreme-channel diagnostics.
- **Runtime**: baseline stage, refinement stage, and end-to-end time.
- **PSNR/SSIM vs historical MATLAB**: only for exact-dimension same-input references.

## Reproducibility metadata

`results/report.json` records the benchmark schema version, source SHA-256, exact source/output shapes, per-image kernel size and its provenance, method configuration, runtime, metrics, Python/NumPy/SciPy/OpenCV versions, Git commit, and UTC generation timestamp.

## Output contract

A full run creates:

```text
results/report.html
results/report.json
results/SUMMARY.md
```

and, for every source, native-resolution `baseline.png`, `annealed_pnp.png`, `extreme_channel.png`, `interim.png`, `kernel.png`, an exact source copy, and compatible historical references when available.

GitHub Actions verifies this contract after the Docker command completes.

## Developer smoke tests

For quick development only:

```bash
python scripts/generate_report.py --limit 1
```

The Docker/CI path never uses `--limit`; it always expects all 23 images.

## Scope of claims

This dataset is suitable for regression, same-input historical fidelity, physical consistency, runtime comparison, and qualitative inspection. A state-of-the-art claim should additionally be evaluated on standard paired deblurring benchmarks under their published protocols.

## Reproduce locally

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts
python -m pytest -q tests
docker compose build test
docker compose run --rm test
```

Open `results/report.html` after completion.
