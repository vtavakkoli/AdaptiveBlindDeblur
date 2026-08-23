# Dataset and historical references

This directory contains the benchmark and historical assets used by the reproducible Docker workflow.

## `dataset/image/`

`dataset/image/` contains **23 source images** originating from the supplied CVPR 2016 dark-channel deblurring release.

The primary benchmark command:

```bash
docker compose run --rm test
```

processes every supported image in this folder at its **original decoded dimensions**.

### Native-resolution rule

The benchmark does not resize, crop, pad-to-a-new-output-size, or otherwise resample a source image for evaluation. Internal algorithmic boundary extension is permitted, but every final restoration must return exactly the source height, width, and channel count.

A shape mismatch is a hard test failure.

## `dataset/results/`

`dataset/results/` contains historical MATLAB/release outputs associated with source filenames.

For a source such as:

```text
dataset/image/real_img2.png
```

the benchmark looks for:

```text
dataset/results/real_img2_result.png
dataset/results/real_img2_kernel.png
```

A historical image is used for PSNR/SSIM comparison only if its decoded dimensions match the source exactly. References are never resized to create a comparison.

These metrics measure **agreement with the historical released output**, not clean-image ground-truth quality.

When a valid historical kernel is available, its odd square dimensions may be used to select the Python benchmark kernel size. The historical kernel values themselves are not supplied to the Python methods.

## Other dataset folders

Other folders under `dataset/` are preserved as research and historical benchmark assets but are not automatically included in the 23-image `dataset/image` Docker benchmark unless the benchmark protocol is explicitly extended.

## Mutability

Docker Compose mounts the complete `dataset/` directory read-only:

```text
./dataset:/app/dataset:ro
```

The benchmark writes only to the generated top-level `results/` directory. This prevents accidental modification of source or historical reference assets.

See [`../docs/BENCHMARKING.md`](../docs/BENCHMARKING.md) for the full evaluation protocol and [`../NOTICE.md`](../NOTICE.md) for provenance/licensing notes.
