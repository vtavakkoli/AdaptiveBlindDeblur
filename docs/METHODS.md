# Methods

This repository compares one classical blind-deblurring baseline with two weight-free research refinements.

The goal is to keep the physical blur model explicit and make each added prior independently testable.

## 1. Dark Channel Baseline (`baseline`)

The baseline follows the optimization structure of Pan et al., *Blind Image Deblurring Using Dark Channel Prior*, CVPR 2016.

For a blurred observation `y`, latent sharp image `x`, and blur kernel `k`, the blind problem can be viewed as alternating between:

1. latent-image estimation under dark-channel and gradient sparsity priors;
2. blur-kernel estimation from salient gradients;
3. kernel pruning, normalization, and centering;
4. coarse-to-fine propagation across image scales;
5. final non-blind TV/L0 restoration using the estimated PSF.

The Python implementation modernizes several low-level operations for speed:

- OpenCV morphology for local minima;
- Numba for remaining local mappings;
- SciPy FFTs for convolution/deconvolution;
- NumPy vectorization for gradients and thresholding;
- float32 iterative buffers.

The objective and update sequence are intended to reproduce the research method structurally, not bit-for-bit MATLAB numerics.

## 2. Annealed Gaussian PnP (`annealed-pnp`)

This is a **diffusion-inspired plug-and-play refinement**, not a trained diffusion model.

It starts from the baseline result `x0` and the same estimated PSF `k`. At each refinement step:

1. sample Gaussian perturbation using a geometrically decreasing noise level;
2. denoise the perturbed estimate with classical non-local means;
3. enforce measurement consistency with the observed blur;
4. retain the deterministic candidate selected by the configured seed/protocol.

The measurement-consistency step solves a quadratic proximal problem of the form:

```text
argmin_x ||Kx - y||² + ρ ||x - z||²
```

where:

- `K` is convolution with the estimated PSF;
- `y` is the observed blurred image;
- `z` is the current denoised prior estimate;
- `ρ` controls the balance between the prior and observed measurement.

Because convolution is diagonal in the Fourier domain, the update is solved efficiently with FFTs.

### Why call it diffusion-inspired?

The method borrows the useful inverse-problem pattern used by diffusion/PnP restoration systems:

- progressively changing noise scale;
- a denoising/image prior;
- repeated data-consistency projection.

However, the current denoiser is NLM rather than a learned score or diffusion network. The repository therefore does **not** describe this method as a diffusion model or as SOTA.

## 3. Extreme-Channel Guided (`extreme-channel`)

Dark-channel assumptions can be weak in bright or saturated image regions. Extreme-channel methods address this general limitation by considering both local dark and local bright evidence.

This repository's lightweight refinement:

1. computes local dark extrema;
2. computes local bright extrema;
3. derives an extreme-confidence map;
4. uses that map to gate detail enhancement;
5. applies explicit FFT measurement consistency after every refinement step.

This is an experimental repository-specific refinement motivated by Extreme Channels Prior research; it is not claimed to reproduce the full optimizer of a specific published ECP method.

## Shared-PSF comparison design

For every source image, only the baseline estimates a blind PSF.

```text
source
  │
  ├── blind DCP baseline ──> estimated PSF k
  │                         │
  │                         ├── Annealed Gaussian PnP
  │                         └── Extreme-Channel Guided
  │
  └── same observed image used by all methods
```

Both refinements use exactly the same `k`.

This design is intentional: if each method estimated an independent PSF, differences in the final image would mix together kernel-estimation differences and restoration-prior differences.

## Historical MATLAB kernels

When `dataset/results/<stem>_kernel.png` exists and is a valid odd square kernel, its **dimensions** may be used as the benchmark kernel size for the Python baseline.

The historical kernel values are never injected into the Python methods.

This improves comparability with the historical release while preserving the blind-estimation nature of the Python benchmark.

## Benchmark profile versus CLI defaults

The full Docker benchmark uses a recorded, bounded iterative profile so all 23 native-resolution images can run reproducibly in CI. The exact parameters are written to `results/report.json`.

The CLI remains independently configurable for deeper single-image experiments.

## Recommended future research extensions

The current architecture intentionally separates prior refinement from data consistency. Natural next steps include:

- replacing NLM with a pretrained diffusion/score denoiser;
- adaptive noise schedules based on estimated blur/noise level;
- ringing-aware candidate selection;
- saturation-aware PSF estimation rather than only post-restoration refinement;
- multi-scale learned priors while preserving explicit blur consistency;
- evaluation on paired GoPro, HIDE, RealBlur, Köhler, and Levin-style benchmarks under their standard protocols.

Any SOTA claim should be made only after such methods are evaluated on established paired benchmarks with accepted distortion and perceptual metrics.

## Primary references

- J. Pan, D. Sun, H. Pfister, M.-H. Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.
- Y. Yan et al., **Image Deblurring via Extreme Channels Prior**, CVPR 2017.

See `CITATION.cff` and the root README for citation guidance.
