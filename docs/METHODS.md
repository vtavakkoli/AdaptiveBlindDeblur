# Methods

This repository contains three current, weight-free restoration paths built around one explicit blur model. The implementation is maintained as an independent experimental framework rather than as a reproduction of a specific published method.

## 1. Adaptive Blind Baseline (`baseline`)

The baseline estimates an unknown point-spread function (PSF) and latent image jointly using a coarse-to-fine alternating optimization.

At each scale it combines:

1. sparse local-minimum/extrema regularization of the latent image;
2. sparse image-gradient regularization;
3. salient-gradient selection for PSF estimation;
4. FFT-domain PSF estimation;
5. PSF non-negativity, pruning, normalization, and centering;
6. propagation of the PSF to the next finer image scale.

After blind estimation, the full-resolution RGB image is restored with TV/L0 regularization and ringing suppression.

The benchmark does **not** use one generic PSF support for every image. `dataset/benchmark_profiles.json` records an explicit support and restoration profile for each of the 23 supplied sources.

### Why image-specific PSF support matters

Blind deblurring is strongly dependent on the maximum motion support that the optimizer is allowed to represent. A 25×25 support cannot represent a long 85- or 115-pixel motion trajectory. If the support is too small, the latent image can retain duplicate edges even when the optimization converges numerically.

The quality benchmark therefore treats PSF support as part of the experiment configuration rather than as a CI shortcut.

## 2. Annealed PnP Refinement (`annealed-pnp`)

This refinement begins from the adaptive baseline and its independently estimated PSF.

Each step performs:

1. Gaussian perturbation at a decreasing noise scale;
2. non-local-means denoising to produce a prior candidate;
3. a closed-form FFT data-consistency update;
4. candidate scoring using blur-model consistency and high-frequency growth;
5. a final artifact-safety blend relative to the baseline.

The data-consistency step has the form:

```text
argmin_x ||Kx - y||² + ρ ||x - z||²
```

where `K` is convolution with the estimated PSF, `y` is the observed image, and `z` is the current prior estimate.

The method is stochastic but reproducible under its seed. It has no learned checkpoint, external neural model, or GPU requirement.

### Artifact-safety guard

A lower reblur error does not automatically mean a better restoration. Blind inverse problems can reduce the residual while creating ringing, duplicated contours, or amplified texture/noise.

The refinement therefore measures:

- improvement in reblur RMSE;
- Laplacian-MAD growth relative to the baseline.

When the candidate adds too much high-frequency energy for the achieved fidelity gain, it is blended back toward the baseline. If it does not improve measurement consistency, the baseline is retained.

## 3. Dual-Extreme Refinement (`extreme-channel`)

This method uses both local dark and local bright extrema to form a spatial confidence map for detail recovery.

For each iteration it:

1. computes local dark extrema;
2. computes local bright extrema;
3. forms a dual-extreme confidence map;
4. applies conservative local-detail enhancement;
5. projects the result back through the blur-model data-consistency step;
6. applies the same final artifact-safety guard used by the annealed refinement.

The refinement is intentionally lightweight and deterministic for a fixed input/configuration.

## Shared PSF design

The comparison uses one independently estimated PSF per source:

```text
observed source
      │
      └── Adaptive Blind Baseline ──> estimated PSF
                      │
                      ├── Annealed PnP Refinement
                      └── Dual-Extreme Refinement
```

Both refinements use the baseline PSF. This means differences between the three displayed restorations come from the restoration stage rather than from three unrelated kernel estimates.

## Benchmark profiles

`dataset/benchmark_profiles.json` contains the quality configuration for every benchmark image:

- `kernel_size`
- `gamma`
- `lambda_dark`
- `lambda_grad`
- `lambda_tv`
- `lambda_l0`
- `weight_ring`

The Docker benchmark requires the profile file to contain exactly one entry for every source image.

Legacy output images and legacy kernel **pixel values** are not used to create the current restorations. They are loaded only after inference for evaluation and visualization.

## Full-quality versus preview mode

The quality benchmark uses:

- five latent/PSF alternations per scale;
- uncapped gradient optimization;
- uncapped local-extrema optimization;
- native image resolution.

The CLI flag `--fast` caps iterative loops only for previews. It must not be used to produce the repository's quality report.

## Design principles

The current implementation prioritizes:

- native-resolution evaluation;
- explicit degradation-model consistency;
- reproducible CPU execution;
- independently estimated PSFs;
- transparent per-image configuration;
- protection against residual-only artifact optimization;
- machine-readable experiment records.

## Future work

Useful extensions include learned denoisers, adaptive PSF-support proposal, spatially varying blur models, saturation-aware kernel estimation, perceptual no-reference metrics, and evaluation on separately licensed paired datasets with known clean targets.
