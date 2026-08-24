# PSF plausibility and support preservation

The adaptive quality pipeline treats the point-spread function (PSF) as a physical object, not only as a numerical variable that minimizes image residuals. The remaining benchmark failures after the MATLAB-equivalent core was restored showed two opposite failure modes:

1. **spurious low-mass branches** — e.g. a long horizontal tail crossing an otherwise plausible motion trajectory;
2. **over-pruned support** — weak but meaningful parts of a curved or compound trajectory disappear before final restoration.

The robust pipeline therefore separates *support preservation* from *structural cleanup*.

## Robust support preservation

Strict parity mode keeps the historical 5%-of-peak PSF threshold and 10% connected-component pruning behavior. Adaptive robust mode instead carries weaker support through the pyramid:

- 2.5%-of-peak iterative support threshold;
- 2.5% connected-component mass threshold;
- 2%-of-peak final threshold before structural validation.

This is intentionally more permissive. It gives long/curved kernels enough opportunity to retain meaningful low-amplitude tails.

## Reference-free plausibility diagnostics

`dark_channel_deblur.psf_quality` computes diagnostics using only the independently estimated PSF:

- **largest connected-component mass** — probability mass of the dominant support;
- **secondary-component mass** — retained disconnected support outside the dominant component;
- **anisotropy** — ratio of weighted principal covariance eigenvalues from the high-confidence PSF core;
- **off-axis mass** — weak support far from the dominant motion direction;
- **weak-line mass** — low-amplitude row/column structures that span an implausibly large fraction of the support.

No legacy result or legacy kernel is used by these diagnostics.

## Structural refinement

`refine_psf_structure()` is deliberately conservative:

1. normalize non-negative PSF mass;
2. retain low-amplitude support down to 1.5% of the peak;
3. remove only tiny disconnected components below 1.2% probability mass;
4. suppress weak long row/column spurs that are far from the high-confidence core;
5. for strongly anisotropic single-motion PSFs only, remove very weak off-axis mass;
6. renormalize the PSF.

Curved and compound trajectories skip the aggressive off-axis step when their core is not strongly anisotropic. This is important for `toy.png` and `wall.png`, where meaningful kernel detail must not collapse into a simplified single line.

## Regression targets

The current visual regression targets are:

- `7_patch_use.png` — remove the weak horizontal branch while preserving the useful curved motion path;
- `toy.png` — preserve weak connected trajectory detail that was previously discarded;
- `wall.png` — preserve complex curved support while keeping the restoration artifact-safe.

The unit suite uses synthetic PSFs to protect these structural properties without learning from or hard-coding legacy kernel pixels.

## Evaluation

The full Docker benchmark may still display legacy kernels side-by-side and report kernel correlation when supports match. Those values are evaluation-only diagnostics. They are not inputs to PSF estimation, plausibility scoring, structural refinement, or candidate selection.
