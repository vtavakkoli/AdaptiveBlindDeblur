# Motion-Trajectory-Constrained PSF Estimation

## Motivation

The existing blind PSF step solves a regularized gradient-domain normal equation over the full two-dimensional kernel support. That is intentionally general, but for motion blur the full `K x K` search space is larger than the physical problem requires. Noise in the latent/blurred gradients can therefore be explained by isolated PSF pixels, side lobes, or disconnected branches.

The experimental `motion-constrained` method adds a physical restriction: a motion PSF must lie on a thin connected camera/object trajectory. The trajectory is **not assumed to be straight**; curved, bent, compound, and self-intersecting paths are allowed.

## Constrained problem

For blurred gradients `b`, latent gradients `l`, and the usual quadratic PSF objective `J(k)`, the free solver estimates a kernel over every pixel in the support. The new solver instead optimizes

```text
minimize    J(k)
subject to  k >= 0
            sum(k) = 1
            support(k) subset M
```

where `M` is a thin connected motion corridor inferred from an unconstrained warm start.

This changes the effective number of admissible PSF variables from approximately `K^2` to the area of a narrow tube around a one-dimensional trajectory.

## How the motion corridor is inferred

1. Obtain a warm-start PSF with the existing normal-equation solver. This warm start is used only to infer structure.
2. Clamp negative values and normalize the warm PSF.
3. Smooth lightly to make weak trajectory fragments easier to connect.
4. Evaluate several low support thresholds and small morphological closing radii.
5. For every connected candidate component, compute a morphological skeleton.
6. Dilate that skeleton by the configured corridor radius.
7. Reconnect nearby skeleton remnants and retain the single connected corridor carrying the most warm-start PSF mass.
8. Choose the corridor that captures high warm-start PSF mass while occupying little of the full kernel support.

The result is one connected admissible region that can follow a curved path without permitting isolated 2-D noise far from the trajectory.

## Projected optimization

After the corridor is fixed for the current blind iteration, projected-gradient updates solve the original quadratic kernel objective. Every iterate is projected onto

```text
{k | k >= 0, sum(k)=1, k outside M = 0}
```

using an exact Euclidean simplex projection on the admissible pixels. Thus non-negativity, unit PSF mass, and the motion-support restriction hold throughout the constrained solve, not only as post-processing.

The step size uses `max(spectrum) + weight` as a conservative upper bound on the normal-operator Lipschitz constant.

## Why this differs from PSF cleanup

`refine_psf_structure()` remains useful as a final structural guard, but it operates after an unconstrained kernel has already influenced the blind image/kernel alternations. The trajectory-constrained method changes the kernel optimization itself, so excluded off-trajectory pixels cannot be used to lower the kernel objective during that solve.

## CLI

```bash
dark-channel-deblur input.png output.png \
  --method motion-constrained \
  --kernel-size 85 \
  --motion-corridor-radius 2 \
  --motion-pgd-steps 24 \
  --kernel-output motion_kernel.png
```

The existing `baseline` path is unchanged. The new mode is experimental so it can be compared directly before deciding whether it should become part of robust default selection.

## Python API

```python
from dark_channel_deblur import DeblurConfig, deblur_image

config = DeblurConfig(
    kernel_size=85,
    kernel_model="motion-trajectory",
    motion_corridor_radius=2,
    motion_pgd_steps=24,
)
result, kernel, interim = deblur_image(image, config)
```

For direct gradient-domain experiments, `estimate_motion_constrained_psf()` is also exported.

## Evidence motivating the experiment

The user-supplied 23-image benchmark report shows many current kernels with isolated/disconnected support relative to the saved legacy kernels. A simple 8%-of-peak connected-component analysis of the report PNGs gave approximately:

- current estimated kernels: **13.7 connected components on average**;
- legacy kernels: **2.35 connected components on average**.

With the final one-connected-corridor rule, corridor inference on those current saved kernels retained about **92% of their PSF mass** while allowing only about **12% of the full kernel area** on average. These figures motivate the reduced search space; they are not a quality claim for the complete deblurring method.

Legacy kernels are **evaluation only**. They are not used to infer the corridor, tune an image-specific path, or provide optimization targets.

## Evaluation plan

The important comparison is not kernel appearance alone. Run the full native-resolution benchmark and compare at least:

- reblur RMSE;
- restoration artifact diagnostics;
- kernel connected-component count / largest-component mass;
- kernel correlation to legacy only as a regression metric;
- final visual ringing / duplicated edges;
- runtime overhead.

A motion prior can be wrong for defocus blur, mixed blur, spatially varying blur, or strongly non-motion degradation. For those cases the unconstrained baseline should remain available.
