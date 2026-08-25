# UGDB experiment: uncertainty-guided Gaussian blind deblurring

This branch introduces an experimental family of blind-restoration refinements designed to test the inference ideas behind a future diffusion-based method without requiring neural checkpoints in CI.

## Motivation

The current learned-weight-free refinements reuse a single PSF estimated by the adaptive blind baseline. UGDB tests whether three additional ideas improve reference-free restoration quality:

1. **Closed-form Gaussian measurement conditioning** instead of a fixed uncalibrated projection.
2. **Null-space-aware prior injection**, so the prior dominates mainly where the blur transfer function is poorly observable.
3. **PSF posterior hypotheses**, obtained by re-estimating a conservative kernel proposal from the current latent image and evaluating several interpolated PSFs.

The full variant also uses two annealed Gaussian perturb/denoise prior samples per iteration. The current NLM prior is deliberately a **diffusion surrogate**, not a claim that a pretrained diffusion model is already integrated. This keeps the first ablation CPU-reproducible and isolates whether the Gaussian/posterior machinery itself is useful.

## Variants

| CLI method | Internal variant | Added mechanism |
|---|---|---|
| `ugdb-linear` | `linear` | Closed-form Gaussian posterior mean, fixed PSF |
| `ugdb-null` | `nullspace` | Gaussian posterior + frequency-wise null-space gating |
| `ugdb-kernel` | `kernel` | Gaussian posterior + latent-driven PSF posterior particles |
| `ugdb-full` | `full` | Null-space gating + PSF posterior + stochastic annealed prior bank |

All variants retain the repository's final artifact/ripple safety guard. Kernel updates are discarded when the restoration falls back to the baseline.

## Gaussian linear update

For observation

```text
y = Kx + n
```

with Gaussian measurement noise and a Gaussian approximation around the current prior estimate `z`, the posterior mean is solved frequency-by-frequency:

```text
X(f) = [K*(f)Y(f)/sigma_n^2(f) + Z(f)/sigma_p^2]
       -------------------------------------------------
       [|K(f)|^2/sigma_n^2(f) + 1/sigma_p^2]
```

The implementation parameterizes `rho = sigma_n^2 / sigma_p^2`, matching the scale of the existing proximal data-consistency update. When PSF uncertainty is zero, the update reduces to the standard closed-form blur-fidelity + prior proximal step.

## Operator uncertainty

For PSF hypotheses `k_1 ... k_M`, UGDB computes their OTFs and estimates a frequency-wise disagreement map:

```text
Var[K(f)] = mean_j |K_j(f) - mean(K(f))|^2
```

The measurement variance is inflated at frequencies where the candidate PSFs disagree. This prevents an uncertain degradation model from being treated as exact.

## Null-space gating

Let

```text
c(f) = |K(f)|^2 / (|K(f)|^2 + tau^2)
```

be a soft observability mask. `ugdb-null` and `ugdb-full` combine the Gaussian posterior with the prior in Fourier space:

```text
X_final(f) = c(f) X_posterior(f) + [1-c(f)] Z(f)
```

Thus the measurement dominates observable frequencies while the prior is protected in poorly conditioned frequencies.

## PSF posterior particles

`ugdb-kernel` and `ugdb-full` estimate a new PSF proposal from gradients of the current latent image using the repository's existing PSF normal-equation solver. A small set of interpolated kernels between the current PSF and the proposal forms a posterior particle bank.

Each image/PSF hypothesis is scored with:

- reblur measurement consistency;
- existing artifact penalties;
- PSF structural plausibility;
- a small drift penalty from the original blind estimate.

A soft evidence weighting forms both the latent-image consensus and the posterior PSF. Kernel tracking is conservative so one uncertain iteration cannot replace the baseline PSF abruptly.

## Run one image

```bash
docker compose run --rm deblur \
  /data/input.png /data/output.png \
  --method ugdb-full \
  --kernel-size 85 \
  --ugdb-steps 4 \
  --ugdb-kernel-hypotheses 4 \
  --seed 7 \
  --kernel-output /data/ugdb_kernel.png
```

## Run the ablation benchmark

Smoke test a few images:

```bash
docker compose run --rm ugdb-experiment --limit 3
```

Run all supplied native-resolution images:

```bash
docker compose run --rm ugdb-experiment
```

Outputs:

```text
results/ugdb_experiment/
├── report.html
├── report.json
├── SUMMARY.md
└── 01_<image>/
    ├── baseline.png
    ├── annealed_pnp.png
    ├── extreme_channel.png
    ├── rgac.png
    ├── ugdb_linear.png
    ├── ugdb_null.png
    ├── ugdb_kernel.png
    └── ugdb_full.png
```

## How a winner is selected

The experiment deliberately does **not** use legacy output pixels as ground truth. Its primary comparison score is the repository's existing reference-free `restoration_score`, which combines reblur RMSE with artifact penalties for excessive noise, high-frequency growth, clipping, and edge amplification. Lower is better.

The report shows both mean score and per-image win count. This is useful for deciding which component deserves further development, but it is not sufficient for a state-of-the-art claim. A publication-quality evaluation should additionally use paired datasets with clean ground truth and report PSNR, SSIM, LPIPS, kernel error, runtime, and neural-function evaluations.

## Next experiment if UGDB wins

If `ugdb-null` or `ugdb-full` consistently beats the current methods, the next PR should replace `_nlm_gaussian_denoiser` with an interchangeable pretrained diffusion/consistency-model prior while preserving exactly the same Gaussian likelihood, operator-uncertainty, and null-space code. That isolates the benefit of the learned prior from the benefit of the new inference mechanism.
