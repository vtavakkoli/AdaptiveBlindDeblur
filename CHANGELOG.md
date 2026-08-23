# Changelog

All notable repository-level changes are documented here.

## 0.3.0 — Native-resolution benchmark

### Changed

- Full `dataset/image` Docker benchmark now processes all 23 sources at native decoded resolution.
- Benchmark no longer resizes inputs or references for CI.
- Output dimensions are validated against source dimensions for every method.
- Docker mounts `dataset/` read-only instead of baking dataset copies into the image.
- Historical `dataset/results/<stem>_result.png` outputs are compared only when dimensions match exactly.
- Historical kernel dimensions can select the Python benchmark kernel size without using historical kernel values as an oracle.
- Report schema upgraded with source SHA-256, environment versions, Git commit, kernel-size provenance, exact shape metadata, and historical-reference status.
- CI now includes Ruff linting, native-resolution contract verification, workflow summary publishing, and a longer benchmark timeout.

### Added

- Native-resolution HTML benchmark report with historical MATLAB cards and kernel comparison where available.
- `results/SUMMARY.md` CI/experiment summary.
- `docs/METHODS.md`.
- `docs/BENCHMARKING.md`.
- `CONTRIBUTING.md`.
- `CITATION.cff`.
- `NOTICE.md`.
- Pull-request template for benchmark/reproducibility review.

## 0.2.0 — Research refinements and full image-folder coverage

- Added Annealed Gaussian PnP refinement.
- Added Extreme-Channel Guided refinement.
- Extended Docker report from one image to all 23 files in `dataset/image`.
- Added per-image and aggregate diagnostics.
- Added Python 3.13-only CI and refinement regression tests.

## 0.1.0 — Initial Python port

- Ported the CVPR 2016 dark-channel blind-deblurring research code from MATLAB to Python.
- Added OpenCV, NumPy, SciPy FFT, and Numba optimizations.
- Added CLI, package API, Docker support, unit tests, and MATLAB regression assets.
