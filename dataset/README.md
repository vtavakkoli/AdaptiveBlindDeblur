# Dataset, profiles, and legacy evaluation assets

This directory contains the source images and experiment configuration used by the reproducible Docker benchmark.

## `dataset/image/`

`dataset/image/` contains **23 source images**. The benchmark processes every supported file at its **original decoded dimensions**.

```bash
docker compose run --rm test
```

### Native-resolution rule

The benchmark does not resize or crop a source for evaluation. Internal algorithmic boundary extension is permitted, but every final restoration must return the exact source height, width, and channel count.

A shape mismatch is a hard test failure.

## `dataset/benchmark_profiles.json`

The quality benchmark has one explicit profile per source image. Profiles contain only algorithm configuration:

- PSF support size;
- gamma;
- sparse-extrema regularization;
- gradient regularization;
- TV final-restoration strength;
- L0 final-restoration strength;
- ringing-removal weight.

The Docker test fails unless the profile keys exactly match the 23 filenames in `dataset/image/`.

This prevents difficult long-motion images from silently falling back to a small generic PSF support.

## `dataset/results/`

`dataset/results/` contains **legacy result and kernel snapshots** retained for regression and visual comparison.

For a source such as:

```text
dataset/image/real_img2.png
```

the report may load:

```text
dataset/results/real_img2_result.png
dataset/results/real_img2_kernel.png
```

These files are evaluation-only:

- legacy result pixels are never passed into current restoration methods;
- legacy kernel pixel values are never passed into current PSF estimation;
- no legacy result is resized to force a comparison;
- PSNR/SSIM vs legacy are regression/fidelity metrics, not clean-image ground-truth scores;
- kernel correlation is shown only when current and legacy supports are directly comparable.

## Other folders

Other folders under `dataset/` are preserved as historical/research assets and are not automatically part of the 23-image Docker benchmark unless the benchmark protocol is explicitly extended.

## Mutability

Docker Compose mounts the complete `dataset/` directory read-only:

```text
./dataset:/app/dataset:ro
```

Generated files are written only to the top-level `results/` directory.

See [`../docs/BENCHMARKING.md`](../docs/BENCHMARKING.md) for the evaluation contract and [`../NOTICE.md`](../NOTICE.md) for asset/provenance notes.
