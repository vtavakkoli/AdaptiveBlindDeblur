from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import fft

from .fft_utils import psf2otf
from .kernel import adjust_psf_center, estimate_psf, prune_kernel, valid_gradients
from .psf_quality import psf_plausibility
from .quality import restoration_score
from .refinement import (
    _artifact_safe_blend,
    _crop,
    _nlm_gaussian_denoiser,
    _pad_for_kernel,
    reblur_image,
)

UGDB_VARIANTS = ("linear", "nullspace", "kernel", "full")


@dataclass(frozen=True, slots=True)
class UGDBDiagnostics:
    """Reference-free diagnostics for the experimental UGDB family."""

    variant: str
    steps: int
    kernel_hypotheses: int
    mean_observable_fraction: float
    mean_kernel_uncertainty: float
    final_score: float
    accepted_kernel_update: bool


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(kernel, dtype=np.float32), 0.0).copy()
    total = float(k.sum())
    if total <= 1e-12:
        k[:] = 0.0
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
        return k
    return (k / total).astype(np.float32)


def _kernel_proposal(
    observed: np.ndarray,
    latent: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    """Estimate a conservative PSF correction from the current latent image."""
    y = _gray(observed)
    x = _gray(latent)
    bx, by = valid_gradients(y)
    lx, ly = valid_gradients(x)
    proposal = estimate_psf(
        bx,
        by,
        lx,
        ly,
        weight=2.0,
        psf_shape=kernel.shape,
        workers=workers,
        peak_fraction=0.02,
    )
    proposal = prune_kernel(proposal, min_component_mass=0.025)
    proposal = adjust_psf_center(proposal)
    return _normalize_kernel(proposal)


def _kernel_hypotheses(
    current: np.ndarray,
    proposal: np.ndarray,
    count: int,
) -> list[np.ndarray]:
    """Build a small posterior particle bank between current and proposed PSFs."""
    if count < 1:
        raise ValueError("kernel_hypotheses must be >= 1")
    current = _normalize_kernel(current)
    proposal = _normalize_kernel(proposal)
    if count == 1:
        return [current]
    alphas = np.linspace(0.0, 0.75, count, dtype=np.float32)
    return [_normalize_kernel((1.0 - float(a)) * current + float(a) * proposal) for a in alphas]


def _spectral_kernel_uncertainty(
    kernels: list[np.ndarray],
    shape: tuple[int, int],
    *,
    workers: int,
) -> tuple[np.ndarray, float]:
    """Approximate frequency-wise operator uncertainty from a PSF particle bank."""
    if len(kernels) <= 1:
        return np.zeros(shape, dtype=np.float32), 0.0
    spectra = np.stack([psf2otf(kernel, shape, workers) for kernel in kernels], axis=0)
    mean_spectrum = np.mean(spectra, axis=0)
    variance = np.mean(np.abs(spectra - mean_spectrum[None, ...]) ** 2, axis=0)
    reference = float(np.quantile(np.abs(mean_spectrum) ** 2, 0.75))
    normalized = np.clip(variance / max(reference, 1e-8), 0.0, 4.0).astype(np.float32)
    return normalized, float(np.mean(normalized))


def gaussian_linear_update(
    observed: np.ndarray,
    prior: np.ndarray,
    kernel: np.ndarray,
    *,
    rho: float,
    noise_variance: float = 1e-4,
    kernel_uncertainty: np.ndarray | None = None,
    nullspace_gate: bool = False,
    null_threshold: float = 0.055,
    workers: int = -1,
) -> tuple[np.ndarray, float]:
    """Closed-form Gaussian posterior mean with optional null-space protection.

    ``rho`` is the equivalent proximal weight sigma_n^2 / sigma_prior^2.  The
    implementation keeps the familiar scale of the existing PnP data-consistency
    update while allowing frequency-dependent measurement uncertainty.
    """
    if rho <= 0:
        raise ValueError("rho must be > 0")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be > 0")
    if null_threshold <= 0:
        raise ValueError("null_threshold must be > 0")

    y = np.asarray(observed, dtype=np.float32)
    z = np.asarray(prior, dtype=np.float32)
    if y.shape != z.shape:
        raise ValueError("observed and prior must have the same shape")

    yp, py, px = _pad_for_kernel(y, kernel)
    zp, _, _ = _pad_for_kernel(z, kernel)
    spatial_shape = yp.shape[:2]
    otf = psf2otf(_normalize_kernel(kernel), spatial_shape, workers)
    y_fft = fft.fft2(yp, axes=(0, 1), workers=workers)
    z_fft = fft.fft2(zp, axes=(0, 1), workers=workers)

    if kernel_uncertainty is None:
        uncertainty = np.zeros(spatial_shape, dtype=np.float32)
    else:
        uncertainty = np.asarray(kernel_uncertainty, dtype=np.float32)
        if uncertainty.shape != spatial_shape:
            raise ValueError("kernel_uncertainty has incompatible shape")
        uncertainty = np.clip(uncertainty, 0.0, 4.0)

    # Inflate measurement variance where competing PSFs disagree.  With zero
    # uncertainty this reduces algebraically to the repository's proximal FFT step.
    effective_noise = float(noise_variance) * (1.0 + 6.0 * uncertainty)
    prior_variance = float(noise_variance) / float(rho)
    h2 = np.abs(otf) ** 2
    denominator = h2 / effective_noise + 1.0 / prior_variance

    if y.ndim == 3:
        otf_view = otf[..., None]
        effective_noise_view = effective_noise[..., None]
        denominator_view = denominator[..., None]
    else:
        otf_view = otf
        effective_noise_view = effective_noise
        denominator_view = denominator

    posterior_fft = (
        np.conj(otf_view) * y_fft / effective_noise_view + z_fft / prior_variance
    ) / denominator_view

    normalized_h2 = h2 / max(float(np.max(h2)), 1e-8)
    observable = normalized_h2 / (normalized_h2 + float(null_threshold) ** 2)
    if nullspace_gate:
        gate = observable[..., None] if y.ndim == 3 else observable
        posterior_fft = gate * posterior_fft + (1.0 - gate) * z_fft

    estimate = fft.ifft2(posterior_fft, axes=(0, 1), workers=workers).real.astype(np.float32)
    estimate = np.clip(_crop(estimate, py, px, y.shape[:2]), 0.0, 1.0)
    return estimate, float(np.mean(observable >= 0.5))


def _candidate_score(
    observed: np.ndarray,
    candidate: np.ndarray,
    kernel: np.ndarray,
    reference_kernel: np.ndarray,
    *,
    workers: int,
) -> float:
    reblurred = reblur_image(candidate, kernel, workers=workers)
    score, _ = restoration_score(observed, candidate, reblurred)
    plausibility = psf_plausibility(kernel)
    drift = float(np.abs(kernel - reference_kernel).sum())
    # The penalties are deliberately small: they stabilize blind selection without
    # forcing the original PSF to win when measurement evidence favors a correction.
    return float(score) + 0.003 * float(plausibility.score) + 0.0008 * drift


def _weighted_consensus(
    candidates: list[np.ndarray],
    kernels: list[np.ndarray],
    scores: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    minimum = float(np.min(values))
    temperature = max(0.0015, abs(minimum) * 0.08)
    weights = np.exp(-(values - minimum) / temperature)
    weights /= max(float(weights.sum()), 1e-12)

    image = np.zeros_like(np.asarray(candidates[0], dtype=np.float32))
    kernel = np.zeros_like(np.asarray(kernels[0], dtype=np.float32))
    for weight, candidate, psf in zip(weights, candidates, kernels, strict=True):
        image += float(weight) * np.asarray(candidate, dtype=np.float32)
        kernel += float(weight) * np.asarray(psf, dtype=np.float32)
    return np.clip(image, 0.0, 1.0), _normalize_kernel(kernel)


def ugdb_restore(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    variant: str = "full",
    steps: int = 4,
    kernel_hypotheses: int = 4,
    sigma_start: float = 0.022,
    sigma_end: float = 0.004,
    seed: int = 0,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, UGDBDiagnostics]:
    """Run an uncertainty-guided Gaussian/diffusion-surrogate blind refinement.

    Variants isolate the proposed contributions:

    * ``linear``: analytic Gaussian measurement conditioning with a fixed PSF.
    * ``nullspace``: linear + frequency-wise null-space gating.
    * ``kernel``: linear + latent-driven PSF posterior hypotheses.
    * ``full``: null-space gating + PSF posterior + stochastic annealed prior samples.

    The current prototype intentionally uses the repository's weight-free NLM
    annealed prior as a diffusion surrogate.  This makes the ablation reproducible
    on CPU and tests the inference mathematics before introducing model weights.
    """
    if variant not in UGDB_VARIANTS:
        raise ValueError(f"variant must be one of {UGDB_VARIANTS}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if kernel_hypotheses < 1:
        raise ValueError("kernel_hypotheses must be >= 1")
    if sigma_start <= 0 or sigma_end <= 0:
        raise ValueError("sigma values must be > 0")

    y = np.asarray(observed, dtype=np.float32)
    base = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0)
    if y.shape != base.shape:
        raise ValueError("observed and initial must have the same shape")

    original_kernel = _normalize_kernel(kernel)
    current_kernel = original_kernel.copy()
    x = base.copy()
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(float(sigma_start), float(sigma_end), int(steps))
    observable_fractions: list[float] = []
    uncertainty_values: list[float] = []

    use_nullspace = variant in {"nullspace", "full"}
    update_kernel = variant in {"kernel", "full"}
    stochastic_prior = variant == "full"

    for index, sigma in enumerate(sigmas):
        rho = 0.055 + 0.105 * index / max(steps - 1, 1)

        if stochastic_prior:
            prior_bank = []
            for _ in range(2):
                noise = rng.normal(0.0, float(sigma), x.shape).astype(np.float32)
                prior_bank.append(_nlm_gaussian_denoiser(np.clip(x + noise, 0.0, 1.0), float(sigma)))
        else:
            prior_bank = [_nlm_gaussian_denoiser(x, float(sigma))]

        if update_kernel:
            proposal = _kernel_proposal(y, x, current_kernel, workers=workers)
            hypotheses = _kernel_hypotheses(current_kernel, proposal, kernel_hypotheses)
        else:
            hypotheses = [current_kernel]

        # All hypotheses use the same reflection-padded support, so a single
        # frequency-wise disagreement map can calibrate their likelihood variance.
        padded, _, _ = _pad_for_kernel(y, current_kernel)
        uncertainty_map, uncertainty_scalar = _spectral_kernel_uncertainty(
            hypotheses,
            padded.shape[:2],
            workers=workers,
        )
        uncertainty_values.append(uncertainty_scalar)

        candidates: list[np.ndarray] = []
        candidate_kernels: list[np.ndarray] = []
        candidate_scores: list[float] = []
        step_observable: list[float] = []
        for prior in prior_bank:
            for hypothesis in hypotheses:
                candidate, observable_fraction = gaussian_linear_update(
                    y,
                    prior,
                    hypothesis,
                    rho=rho,
                    kernel_uncertainty=uncertainty_map,
                    nullspace_gate=use_nullspace,
                    workers=workers,
                )
                score = _candidate_score(
                    y,
                    candidate,
                    hypothesis,
                    original_kernel,
                    workers=workers,
                )
                candidates.append(candidate)
                candidate_kernels.append(hypothesis)
                candidate_scores.append(score)
                step_observable.append(observable_fraction)

        x, posterior_kernel = _weighted_consensus(candidates, candidate_kernels, candidate_scores)
        observable_fractions.append(float(np.mean(step_observable)))
        if update_kernel:
            # Conservative posterior tracking prevents one noisy latent iteration
            # from replacing a stable blind estimate in a single jump.
            current_kernel = _normalize_kernel(0.72 * current_kernel + 0.28 * posterior_kernel)

    safe = _artifact_safe_blend(y, base, x, current_kernel, workers=workers)
    accepted_kernel_update = bool(update_kernel and not np.allclose(current_kernel, original_kernel))
    if np.allclose(safe, base, atol=1e-7, rtol=0.0):
        current_kernel = original_kernel
        accepted_kernel_update = False

    final_reblur = reblur_image(safe, current_kernel, workers=workers)
    final_score, _ = restoration_score(y, safe, final_reblur)
    diagnostics = UGDBDiagnostics(
        variant=variant,
        steps=int(steps),
        kernel_hypotheses=int(len(hypotheses)),
        mean_observable_fraction=float(np.mean(observable_fractions)),
        mean_kernel_uncertainty=float(np.mean(uncertainty_values)),
        final_score=float(final_score),
        accepted_kernel_update=accepted_kernel_update,
    )
    return safe.astype(np.float32), current_kernel.astype(np.float32), diagnostics


def ugdb_refine(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    variant: str = "full",
    steps: int = 4,
    kernel_hypotheses: int = 4,
    sigma_start: float = 0.022,
    sigma_end: float = 0.004,
    seed: int = 0,
    workers: int = -1,
) -> np.ndarray:
    """Convenience wrapper returning only the restored image."""
    restored, _, _ = ugdb_restore(
        observed,
        initial,
        kernel,
        variant=variant,
        steps=steps,
        kernel_hypotheses=kernel_hypotheses,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        seed=seed,
        workers=workers,
    )
    return restored
