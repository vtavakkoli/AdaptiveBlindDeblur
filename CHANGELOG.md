# Changelog

All notable repository-level changes are documented here.

## 0.4.0 — Full-quality adaptive benchmark

### Fixed

- Removed the generic small-PSF benchmark shortcut that could leave long motion blur unresolved.
- Added explicit per-image benchmark profiles, including large PSF supports for difficult motion cases.
- Removed optimization-loop caps from the quality benchmark; `--fast` remains preview-only.
- Added artifact-safety guards so refinements cannot be accepted solely by lowering reblur residual while strongly amplifying high-frequency noise/ringing.
- Updated Docker/CI validation to require estimated PSF dimensions to match the recorded benchmark profile.

### Changed

- Repositioned the project as an independent experimental deblurring framework rather than a port/reproduction of an older paper.
- Renamed report-facing methods to Adaptive Blind Baseline, Annealed PnP Refinement, and Dual-Extreme Refinement.
- Renamed previous saved outputs/kernels as legacy evaluation assets.
- Legacy result pixels and legacy kernel values are explicitly evaluation-only and never inference inputs.
- Package version bumped to 0.4.0.

### Added

- `dataset/benchmark_profiles.json` with one quality profile for every source image.
- Regression tests for profile coverage, long-motion PSF support, and bounded refinement artifact growth.

## 0.3.0 — Native-resolution benchmark

- Full Docker benchmark processes all 23 sources at native decoded resolution.
- Output dimensions are validated for every method.
- Docker mounts `dataset/` read-only.
- Added exact-shape legacy comparison, machine-readable experiment metadata, Ruff linting, `SUMMARY.md`, methods/benchmark docs, contribution guidance, citation metadata, notice, and PR template.

## 0.2.0 — Refinements and full image-folder coverage

- Added annealed stochastic plug-and-play refinement.
- Added dual-extreme local-detail refinement.
- Extended Docker report from one image to all 23 files in `dataset/image`.
- Added per-image and aggregate diagnostics.
- Added Python 3.13-only CI and refinement regression tests.

## 0.1.0 — Initial implementation

- Added multi-scale blind PSF estimation, TV/L0 restoration, CLI, package API, Docker support, unit tests, and regression assets.
