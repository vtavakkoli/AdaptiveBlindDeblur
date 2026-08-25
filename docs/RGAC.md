# Residual-Guided Adaptive Consensus (RGAC)

RGAC is a **reference-free restoration method** for the AdaptiveBlindDeblur pipeline. It is not a trained neural model and contains no learned weights. It consumes the observed blurred image, the independently estimated PSF, and the robust baseline restoration, then combines complementary restoration priors only when the resulting image remains physically and artifact-wise plausible.

## Motivation

The benchmark methods have complementary behavior:

- the adaptive robust baseline is usually the safest and lowest-noise reconstruction;
- Annealed PnP can improve denoising and local consistency;
- Dual-Extreme can recover stronger text/edge detail but may amplify noise;
- conservative data-consistent restoration is useful when the PSF is uncertain or ringing-prone.

A single global refinement therefore cannot dominate every image region. RGAC makes the decision **locally** and then performs one final global blur-consistency projection.

## Candidate set

For one observed image `y` and estimated PSF `k`, RGAC forms four candidates:

1. **baseline** — the adaptive robust restoration;
2. **conservative** — bilateral prior + closed-form blur data consistency;
3. **annealed_pnp** — the existing guarded Annealed PnP candidate;
4. **extreme_channel** — the existing guarded Dual-Extreme candidate.

Benchmark execution reuses the already-computed PnP and Dual-Extreme images, so RGAC does not duplicate those expensive stages.

## PSF confidence

`psf_plausibility()` provides reference-free structural diagnostics. RGAC converts largest-component mass, weak-line energy, off-axis energy, anisotropy, and excessive disconnected support into a bounded PSF confidence.

When PSF confidence is low, RGAC automatically prefers conservative candidates and uses a stronger consensus prior during the final data-consistency step. When confidence is high, detail-producing candidates are allowed more influence.

## Local energy

For every candidate, RGAC computes spatial maps for:

- local reblur residual;
- excess edge amplification relative to the observation;
- excess high-frequency/Laplacian energy;
- clipping growth;
- a global reference-free restoration score.

These terms form a local candidate energy. Candidate weights are obtained with a temperature-controlled softmax and spatial Gaussian smoothing. Smooth weighting is used instead of hard pixel selection to avoid seams.

Conceptually,

```text
E_j(p) = residual + edge penalty + high-pass penalty + clipping penalty + global prior
W_j(p) = softmax(-E_j(p) / T)
x_consensus(p) = sum_j W_j(p) * x_j(p)
```

## Final blur consistency and safety

The fused consensus is projected through the same closed-form blur-consistency proximal update used by the refinement pipeline. RGAC then evaluates the full proposal and several conservative blends against the baseline.

A proposal is accepted only when:

- the reference-free restoration score improves;
- the result is not classified as ripple-risky;
- clipping, noise, high-pass energy, and edge growth remain inside safety budgets.

Otherwise RGAC returns the robust baseline. This makes the new method fail-safe with respect to the repository's own reference-free quality objective.

## API

```python
from dark_channel_deblur import residual_guided_adaptive_consensus_refine

rgac = residual_guided_adaptive_consensus_refine(
    observed,
    baseline,
    kernel,
    workers=-1,
)
```

For benchmark callers that already computed the refinement candidates:

```python
rgac, diagnostics = residual_guided_adaptive_consensus_refine(
    observed,
    baseline,
    kernel,
    annealed=annealed,
    extreme=extreme,
    return_diagnostics=True,
)
```

## CLI

```bash
dark-channel-deblur input.png output.png \
  --method rgac \
  --kernel-size 85
```

## Scientific status

RGAC is an experimental algorithm introduced in this repository. A benchmark improvement should be claimed only after the complete 23-image native-resolution benchmark has finished and the visual outputs have been inspected. Legacy images and kernels remain evaluation-only and are never used to compute RGAC weights or select candidates.
