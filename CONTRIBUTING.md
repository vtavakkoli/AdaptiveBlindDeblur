# Contributing

Contributions are welcome when they preserve the repository's reproducibility and benchmark integrity.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

Run the fast quality gates before opening a pull request:

```bash
python -m ruff check src tests scripts
python -m pytest -q tests
```

For benchmark-affecting changes, also run:

```bash
docker compose build test
docker compose run --rm test
```

## Benchmark invariants

Changes to deblurring, report generation, Docker, or dataset handling must preserve these rules:

1. `dataset/image/` sources are processed at native resolution.
2. Final outputs must have exactly the same dimensions as their source.
3. Historical references are never resized to make a comparison possible.
4. PSNR/SSIM against `dataset/results/` must be labeled as historical-output agreement, not ground truth.
5. The two research refinements reuse the same baseline-estimated PSF within each image.
6. Dataset assets are mounted read-only in Docker.
7. Generated experiment files belong in `results/` and must not be committed accidentally.

See `docs/BENCHMARKING.md` for the full protocol.

## Pull request checklist

A benchmark-related PR should explain:

- what algorithmic behavior changed;
- whether the benchmark schema changed;
- whether runtime or memory use changed materially;
- whether any metric definition changed;
- whether output dimensions remain identical to source dimensions;
- whether new dependencies or model weights were introduced;
- how the change was validated.

If a new method is added, include focused unit tests and update `docs/METHODS.md`.

## Scientific claims

Please distinguish clearly between:

- a repository-specific experimental variant;
- a faithful implementation of a published method;
- a learned model using published weights;
- a state-of-the-art claim supported by standard benchmark evidence.

Do not label a weight-free denoising heuristic as a trained diffusion model.

## Dataset and reference assets

Do not replace or recompress existing dataset/reference files solely to reduce CI runtime. If benchmark runtime must be reduced, optimize the algorithm or create an explicitly separate smoke-test profile; do not silently change the official native-resolution evaluation.

## Code style

- Python 3.13+
- type annotations for public/new functions where practical
- deterministic tests
- clear docstrings for numerical routines
- avoid unnecessary dependencies
- prefer NumPy/OpenCV/SciPy vectorization over Python pixel loops

## Commit scope

Keep algorithm, benchmark-protocol, and dataset-asset changes logically separated when possible. Generated `results/` artifacts are CI outputs and are ignored by Git.
