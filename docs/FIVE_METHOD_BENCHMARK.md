# Five-Method Docker Benchmark and PSF Report

This benchmark extends the full native-resolution Docker test to compare five methods on all 23 configured source images:

1. Adaptive Robust Baseline
2. Motion-Constrained
3. Annealed PnP
4. Dual-Extreme
5. Residual-Guided Adaptive Consensus (RGAC)

## Run

Cleanly regenerate the complete benchmark and report:

```bash
docker compose run --rm test --rebuild
```

`--rebuild` is a project-level argument accepted by `scripts/run_docker_test.py`; it requests a clean regeneration of the benchmark outputs. If the Docker image itself also needs to be rebuilt, use Compose's build option before the service name:

```bash
docker compose run --rm --build test --rebuild
```

## Outputs

The run produces:

```text
results/
├── report.html
├── report.json
├── SUMMARY.md
└── images/
    └── <case>/
        ├── baseline.png
        ├── baseline_kernel.png
        ├── motion_constrained.png
        ├── motion_constrained_kernel.png
        ├── annealed_pnp.png
        ├── annealed_pnp_kernel.png
        ├── extreme_channel.png
        ├── extreme_channel_kernel.png
        ├── rgac.png
        ├── rgac_kernel.png
        ├── interim.png
        └── interim_motion.png
```

That is **23 × 5 = 115 restorations** and **115 PSF visualizations**.

## What the five kernels mean

The kernel panels are intentionally explicit about their role.

- **Baseline kernel** is the operational PSF estimated by the existing free 2-D blind optimizer.
- **Motion-Constrained kernel** is the operational PSF estimated under the connected motion-trajectory restriction.
- **Annealed PnP, Dual-Extreme, and RGAC kernels** are diagnostic PSF refits from each method's final latent image using the same gradient-domain PSF estimator. They are generated only for comparison and are **not fed back into restoration**.

This preserves the actual algorithms while making it possible to inspect whether a final latent image implies a cleaner, less fragmented, more physically plausible motion kernel.

## Winner selection

The report identifies a per-image winner and an overall winner using only reference-free measurements. Legacy images and legacy kernels do not participate in winner selection.

For each image, four terms are min-max normalized across the five methods and combined as:

```text
reference-free score =
    0.60 × guarded blind restoration score
  + 0.20 × diagnostic reblur RMSE
  + 0.15 × PSF plausibility penalty
  + 0.05 × PSF fragmentation
```

Lower is better.

The report also shows legacy PSNR/SSIM and legacy-kernel correlation when shapes are directly comparable, but these are regression diagnostics only; legacy assets are not ground truth.

## Report layout

`results/report.html` includes:

- an overall reference-free winner panel;
- aggregate win count and mean score for all five methods;
- restoration metrics and runtime;
- mean PSF connected-component count;
- mean PSF plausibility penalty;
- legacy-kernel correlation when comparable;
- all 23 image cases;
- five restoration images per case;
- five kernel images per case;
- a clear per-image winner banner;
- explanation of which kernels are operational and which are diagnostic refits.

The purpose is to make the motion-constrained experiment visually and numerically auditable rather than selecting a method from kernel appearance alone.
