from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .psf_quality import psf_plausibility
from .quality import artifact_diagnostics, restoration_score, ripple_risk
from .refinement import (
    _data_consistency,
    annealed_pnp_refine,
    extreme_channel_refine,
    reblur_image,
)


@dataclass(frozen=True, slots=True)
class RGACDiagnostics:
    """Reference-free diagnostics produced by Residual-Guided Adaptive Consensus."""

    psf_confidence: float
    candidate_scores: dict[str, float]
    mean_weights: dict[str, float]
    accepted_variant: str


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return (
        arr[..., 0] * np.float32(0.298936021293775)
        + arr[..., 1] * np.float32(0.587043074451121)
        + arr[..., 2] * np.float32(0.114020904255103)
    ).astype(np.float32)


def _smooth_map(values: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    return cv2.GaussianBlur(
        np.asarray(values, dtype=np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    g = _gray(image)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _highpass_magnitude(image: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(_gray(image), cv2.CV_32F, ksize=3)
    return np.abs(lap).astype(np.float32)


def _clipping_map(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    clipped = (arr <= 0.003) | (arr >= 0.997)
    if clipped.ndim == 3:
        clipped = np.mean(clipped.astype(np.float32), axis=2)
    else:
        clipped = clipped.astype(np.float32)
    return _smooth_map(clipped, sigma=2.0)


def _psf_confidence(kernel: np.ndarray) -> float:
    """Convert structural PSF plausibility into a conservative [0.15, 0.98] confidence."""
    diag = psf_plausibility(kernel)
    confidence = (
        0.28
        + 0.72 * float(diag.largest_component_mass)
        - 1.80 * float(diag.score)
        - 0.30 * max(0.0, float(diag.secondary_component_mass) - 0.45)
    )
    return float(np.clip(confidence, 0.15, 0.98))


def _conservative_candidate(
    observed: np.ndarray,
    baseline: np.ndarray,
    kernel: np.ndarray,
    *,
    psf_confidence: float,
    workers: int,
) -> np.ndarray:
    """Create a low-ringing candidate without another blind-kernel solve."""
    base = np.clip(np.asarray(baseline, dtype=np.float32), 0.0, 1.0)
    sigma_color = 0.030 + 0.025 * (1.0 - psf_confidence)
    prior = cv2.bilateralFilter(base, d=5, sigmaColor=sigma_color, sigmaSpace=2.0)
    rho = 0.14 + 0.16 * (1.0 - psf_confidence)
    return _data_consistency(observed, prior, kernel, rho, workers=workers)


def _candidate_maps(
    observed: np.ndarray,
    candidate: np.ndarray,
    kernel: np.ndarray,
    *,
    psf_confidence: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return local fidelity/artifact maps plus the global blind restoration score."""
    y = np.asarray(observed, dtype=np.float32)
    x = np.clip(np.asarray(candidate, dtype=np.float32), 0.0, 1.0)
    reblurred = reblur_image(x, kernel, workers=workers)
    diff = reblurred - y
    if diff.ndim == 3:
        residual = np.sqrt(np.mean(diff * diff, axis=2)).astype(np.float32)
    else:
        residual = np.abs(diff).astype(np.float32)
    residual = _smooth_map(residual, sigma=1.5)

    observed_edge = _smooth_map(_gradient_magnitude(y), sigma=1.2)
    candidate_edge = _smooth_map(_gradient_magnitude(x), sigma=1.2)
    edge_ratio = (candidate_edge + 0.004) / (observed_edge + 0.004)
    edge_allowance = 1.24 + 0.36 * psf_confidence
    edge_penalty = np.clip((edge_ratio - edge_allowance) / 1.25, 0.0, 3.0).astype(np.float32)

    observed_hp = _smooth_map(_highpass_magnitude(y), sigma=1.2)
    candidate_hp = _smooth_map(_highpass_magnitude(x), sigma=1.2)
    hp_ratio = (candidate_hp + 0.003) / (observed_hp + 0.003)
    hp_allowance = 1.28 + 0.42 * psf_confidence
    highpass_penalty = np.clip((hp_ratio - hp_allowance) / 1.35, 0.0, 3.0).astype(np.float32)

    clipping_penalty = np.maximum(_clipping_map(x) - _clipping_map(y), 0.0).astype(np.float32)
    global_score, _ = restoration_score(y, x, reblurred)
    return residual, edge_penalty, highpass_penalty, clipping_penalty, float(global_score)


def _soft_consensus(
    observed: np.ndarray,
    candidates: list[np.ndarray],
    names: list[str],
    kernel: np.ndarray,
    *,
    psf_confidence: float,
    workers: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    maps = [
        _candidate_maps(
            observed,
            candidate,
            kernel,
            psf_confidence=psf_confidence,
            workers=workers,
        )
        for candidate in candidates
    ]

    residuals = np.stack([item[0] for item in maps], axis=0)
    local_best = np.min(residuals, axis=0)
    local_scale = np.maximum(np.median(residuals, axis=0), 0.004)
    residual_energy = np.clip(
        (residuals - local_best[None, ...]) / local_scale[None, ...],
        0.0,
        4.0,
    )

    scores = np.asarray([item[4] for item in maps], dtype=np.float32)
    score_min = float(np.min(scores))
    score_scale = max(abs(score_min), 0.01)
    global_energy = np.clip((scores - score_min) / score_scale, 0.0, 4.0)

    energies: list[np.ndarray] = []
    uncertainty = 1.0 - psf_confidence
    method_bias = {
        "baseline": 0.0,
        "conservative": -0.10 * uncertainty,
        "annealed_pnp": 0.06 * uncertainty,
        "extreme_channel": 0.09 * uncertainty,
    }
    for index, name in enumerate(names):
        _, edge_penalty, hp_penalty, clip_penalty, _ = maps[index]
        energy = (
            1.75 * residual_energy[index]
            + 0.80 * edge_penalty
            + 1.00 * hp_penalty
            + 0.70 * clip_penalty
            + 0.22 * global_energy[index]
            + float(method_bias.get(name, 0.0))
        )
        energies.append(energy.astype(np.float32))

    stack = np.stack(energies, axis=0)
    stack -= np.min(stack, axis=0, keepdims=True)
    temperature = 0.42 + 0.20 * uncertainty
    weights = np.exp(-stack / max(temperature, 1e-4)).astype(np.float32)
    for index in range(weights.shape[0]):
        weights[index] = _smooth_map(weights[index], sigma=2.2)
    weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-8)

    first = np.asarray(candidates[0], dtype=np.float32)
    consensus = np.zeros_like(first, dtype=np.float32)
    for index, candidate in enumerate(candidates):
        weight = weights[index]
        if first.ndim == 3:
            weight = weight[..., None]
        consensus += weight * np.asarray(candidate, dtype=np.float32)

    candidate_scores = {name: float(score) for name, score in zip(names, scores, strict=True)}
    mean_weights = {name: float(np.mean(weights[index])) for index, name in enumerate(names)}
    return np.clip(consensus, 0.0, 1.0), candidate_scores, mean_weights


def _safe_candidate(observed: np.ndarray, candidate: np.ndarray, kernel: np.ndarray) -> bool:
    diag = artifact_diagnostics(observed, candidate)
    if ripple_risk(diag, kernel_size=int(max(kernel.shape))):
        return False
    return bool(
        diag.clipping_growth <= 0.055
        and diag.noise_ratio <= 2.35
        and diag.highpass_ratio <= 3.10
        and diag.edge_ratio <= 2.70
    )


def _accept_consensus(
    observed: np.ndarray,
    baseline: np.ndarray,
    projected: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, str]:
    """Keep RGAC only when its reference-free score improves without ripple risk."""
    y = np.asarray(observed, dtype=np.float32)
    base = np.clip(np.asarray(baseline, dtype=np.float32), 0.0, 1.0)
    proposal = np.clip(np.asarray(projected, dtype=np.float32), 0.0, 1.0)

    base_reblur = reblur_image(base, kernel, workers=workers)
    base_score, _ = restoration_score(y, base, base_reblur)
    best = base
    best_score = float(base_score)
    best_name = "baseline_fallback"

    for alpha in (1.00, 0.85, 0.70, 0.55, 0.40):
        candidate = np.clip(base + alpha * (proposal - base), 0.0, 1.0).astype(np.float32)
        if not _safe_candidate(y, candidate, kernel):
            continue
        reblurred = reblur_image(candidate, kernel, workers=workers)
        score, _ = restoration_score(y, candidate, reblurred)
        if float(score) < best_score * 0.999:
            best = candidate
            best_score = float(score)
            best_name = "consensus" if alpha == 1.0 else f"consensus_blend_{alpha:.2f}"

    return best.astype(np.float32), best_name


def residual_guided_adaptive_consensus_refine(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    annealed: np.ndarray | None = None,
    extreme: np.ndarray | None = None,
    seed: int = 0,
    workers: int = -1,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, RGACDiagnostics]:
    """Residual-Guided Adaptive Consensus (RGAC) restoration.

    RGAC is a reference-free multi-prior restoration method rather than a trained
    model. It builds four complementary candidates from one blind PSF: the robust
    baseline, a conservative data-consistent candidate, Annealed PnP, and
    Dual-Extreme refinement. Per-pixel confidence maps penalize reblur residual,
    excess edge/high-frequency energy, and clipping. Smooth softmax weights fuse
    the candidates, then a final blur-consistency proximal step and global ripple
    guard ensure that consensus cannot silently replace a safer baseline with a
    worse artifact-prone solution.

    ``annealed`` and ``extreme`` can be supplied by benchmark callers to reuse
    already-computed candidates and avoid duplicate expensive refinement work.
    Given identical inputs and ``seed``, RGAC is reproducible and uses no learned
    parameters or legacy-reference pixels.
    """
    y = np.asarray(observed, dtype=np.float32)
    base = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0)
    k = np.asarray(kernel, dtype=np.float32)
    if y.shape != base.shape:
        raise ValueError("observed and initial must have the same shape")
    if k.ndim != 2 or min(k.shape) < 3:
        raise ValueError("kernel must be a 2-D PSF with support >= 3")
    if not np.isfinite(y).all() or not np.isfinite(base).all() or not np.isfinite(k).all():
        raise ValueError("RGAC inputs must contain only finite values")

    confidence = _psf_confidence(k)
    conservative = _conservative_candidate(
        y,
        base,
        k,
        psf_confidence=confidence,
        workers=workers,
    )
    pnp = (
        np.asarray(annealed, dtype=np.float32)
        if annealed is not None
        else annealed_pnp_refine(y, base, k, seed=seed, workers=workers)
    )
    dual = (
        np.asarray(extreme, dtype=np.float32)
        if extreme is not None
        else extreme_channel_refine(y, base, k, workers=workers)
    )
    for name, candidate in (("annealed", pnp), ("extreme", dual)):
        if candidate.shape != y.shape:
            raise ValueError(f"{name} candidate must have the same shape as observed")

    names = ["baseline", "conservative", "annealed_pnp", "extreme_channel"]
    candidates = [base, conservative, pnp, dual]
    consensus, candidate_scores, mean_weights = _soft_consensus(
        y,
        candidates,
        names,
        k,
        psf_confidence=confidence,
        workers=workers,
    )

    # A less plausible PSF receives a stronger consensus prior so uncertain blur
    # physics cannot overrule the conservative multi-prior estimate.
    rho = 0.10 + 0.18 * (1.0 - confidence)
    projected = _data_consistency(y, consensus, k, rho, workers=workers)
    result, accepted = _accept_consensus(y, base, projected, k, workers=workers)

    if not return_diagnostics:
        return result
    diagnostics = RGACDiagnostics(
        psf_confidence=confidence,
        candidate_scores=candidate_scores,
        mean_weights=mean_weights,
        accepted_variant=accepted,
    )
    return result, diagnostics
