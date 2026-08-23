# Contributing

Contributions are welcome when they preserve reproducibility and benchmark integrity.

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

Changes to restoration, report generation, Docker, profiles, or dataset handling must preserve these rules:

1. `dataset/image/` sources are processed at native resolution.
2. Final outputs have exactly the same dimensions as their source.
3. `dataset/benchmark_profiles.json` contains exactly one valid profile for every benchmark source.
4. The estimated PSF support matches the configured profile for that source.
5. Legacy references are evaluation-only and are never resized to force a comparison.
6. Legacy result pixels and legacy kernel values are never supplied to current restoration methods.
7. PSNR/SSIM against `dataset/results/` are labeled as legacy-output agreement, not ground truth.
8. The two refinements reuse the same independently estimated baseline PSF within each case.
9. Dataset assets are mounted read-only in Docker.
10. Generated experiment files belong in `results/` and must not be committed accidentally.

See `docs/BENCHMARKING.md` for the complete protocol.

## Pull request checklist

A benchmark-related PR should explain:

- what algorithmic behavior changed;
- whether a benchmark profile changed and why;
- whether the benchmark schema changed;
- whether runtime or memory use changed materially;
- whether any metric definition changed;
- whether output dimensions remain identical to source dimensions;
- whether new dependencies or model weights were introduced;
- how artifact/ringing behavior was checked;
- how the change was validated.

If a new method is added, include focused unit tests and update `docs/METHODS.md`.

## Method and quality claims

Describe methods according to what the current code actually implements. Do not claim that this repository reproduces a specific published algorithm unless such fidelity is intentionally re-established and independently validated.

Do not call a classical or weight-free denoising heuristic a trained diffusion model, and do not make a state-of-the-art claim without an appropriate paired benchmark protocol.

## Dataset and legacy assets

Do not replace or recompress existing source/legacy files solely to reduce CI runtime. If runtime needs improvement, optimize the computation or introduce an explicitly separate smoke-test workflow; do not silently weaken the native-resolution quality benchmark.

## Code style

- Python 3.13+
- type annotations for public/new functions where practical
- deterministic tests
- clear docstrings for numerical routines
- avoid unnecessary dependencies
- prefer NumPy/OpenCV/SciPy vectorization over Python pixel loops

## Commit scope

Keep algorithm, benchmark-protocol, and dataset-asset changes logically separated when possible. Generated `results/` artifacts are CI outputs and are ignored by Git.
