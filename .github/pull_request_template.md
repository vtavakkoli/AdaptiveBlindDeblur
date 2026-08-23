## Summary

Describe the change and why it is needed.

## Change type

- [ ] Algorithm / numerical method
- [ ] Benchmark / metrics / report
- [ ] Tests / CI / Docker
- [ ] Documentation / metadata
- [ ] Dataset or reference assets

## Reproducibility impact

- [ ] Source images remain native resolution in the official benchmark.
- [ ] Final output dimensions remain exactly equal to source dimensions.
- [ ] Historical references are not resized or otherwise altered for comparison.
- [ ] Same-input historical PSNR/SSIM is labeled as fidelity, not ground truth.
- [ ] Both research refinements reuse the baseline-estimated PSF.
- [ ] New parameters/dependencies are recorded in documentation/report metadata.

If any item above is intentionally changed, explain why and update `docs/BENCHMARKING.md`.

## Validation

- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest -q tests`
- [ ] `docker compose build test`
- [ ] `docker compose run --rm test`
- [ ] `results/report.html` inspected for representative cases

## Performance impact

Describe expected runtime, memory, or artifact-size changes.

## Scientific positioning

If this PR adds or changes a research method, state whether it is:

- a repository-specific experimental variant;
- a reimplementation of a published method;
- a learned method using external weights;
- or a benchmark-backed performance claim.

Include citations where appropriate.
