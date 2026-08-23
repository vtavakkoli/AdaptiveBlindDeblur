# `real_img2` legacy regression example

This directory preserves a compact same-input regression snapshot from an earlier version of the project/data package. It is retained for compatibility and historical inspection; the current full-quality benchmark lives under `dataset/` and `results/`.

## Files

- `input_compare.jpg` — compact preview of the observed source.
- `matlab_reference/result_compare.jpg` — legacy saved-result preview. The directory name is retained for compatibility.
- `matlab_reference/kernel.png` — legacy saved kernel image.
- `python/full_result_compare.jpg` and `full_kernel.png` — earlier Python full-mode snapshots.
- `python/fast_result_compare.jpg` and `fast_kernel.png` — earlier preview-mode snapshots.
- `metrics.json` — historical agreement measurements computed before preview encoding.

## Interpretation

The values in `metrics.json` compare earlier Python snapshots with the legacy saved result. They are **not ground-truth restoration scores** and they do not define the current v0.4 quality benchmark.

Current development should use:

```bash
docker compose run --rm test
```

which processes all 23 `dataset/image/` sources at native resolution with explicit profiles and artifact-guarded refinements.
