from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import fft

from .fft_utils import psf2otf
from .quality import (
    artifact_diagnostics,
    clipping_fraction,
    edge_energy,
    highpass_rms,
    noise_mad,
    restoration_score,
    ripple_risk,
)


def _pad_for_kernel(image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, int, int]:
    py = kernel.shape[0] // 2
    px = kernel.shape[1] // 2
    padded = cv2.copyMakeBorder(
        np.asarray(image, dtype=np.float32),
        py,
        py,
        px,
        px,
        cv2.BORDER_REFLECT_101,
    )
    return padded, py, px


def _crop(image: np.ndarray, py: int, px: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return image[py : py + h, px : px + w, ...]


def reblur_image(image: np.ndarray, kernel: np.ndarray, *, workers: int = -1) -> np.ndarray:
    """Apply a PSF with reflection padding for measurement-consistency diagnostics."""
    arr = np.asarray(image, dtype=np.float32)
    padded, py, px = _pad_for_kernel(arr, kernel)
    otf = psf2otf(np.asarray(kernel, dtype=np.float32), padded.shape[:2], workers)
    spectrum = fft.fft2(padded, axes=(0, 1), workers=workers)
    if arr.ndim == 3:
        otf = otf[..., None]
    blurred = fft.ifft2(otf * spectrum, axes=(0, 1), workers=workers).real.astype(np.float32)
    return np.clip(_crop(blurred, py, px, arr.shape[:2]), 0.0, 1.0)


def _data_consistency(
    observed: np.ndarray,
    prior: np.ndarray,
    kernel: np.ndarray,
    rho: float,
    *,
    workers: int = -1,
) -> np.ndarray:
    """Closed-form proximal step for blur fidelity plus a restoration prior."""
    if rho <= 0:
        raise ValueError("rho must be > 0")
    y = np.asarray(observed, dtype=np.float32)
    z = np.asarray(prior, dtype=np.float32)
    if y.shape != z.shape:
        raise ValueError("observed and prior must have the same shape")

    yp, py, px = _pad_for_kernel(y, kernel)
    zp, _, _ = _pad_for_kernel(z, kernel)
    otf = psf2otf(np.asarray(kernel, dtype=np.float32), yp.shape[:2], workers)
    y_fft = fft.fft2(yp, axes=(0, 1), workers=workers)
    z_fft = fft.fft2(zp, axes=(0, 1), workers=workers)
    denominator = np.abs(otf) ** 2 + float(rho)
    if y.ndim == 3:
        otf = otf[..., None]
        denominator = denominator[..., None]
    estimate = fft.ifft2(
        (np.conj(otf) * y_fft + float(rho) * z_fft) / denominator,
        axes=(0, 1),
        workers=workers,
    ).real.astype(np.float32)
    return np.clip(_crop(estimate, py, px, y.shape[:2]), 0.0, 1.0)


def _nlm_gaussian_denoiser(image: np.ndarray, sigma: float) -> np.ndarray:
    """CPU-friendly denoiser used as a plug-and-play image prior."""
    arr = np.asarray(image, dtype=np.float32)
    u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    h = max(2.0, min(12.0, float(sigma) * 255.0 * 0.55))
    if u8.ndim == 2:
        out = cv2.fastNlMeansDenoising(u8, None, h, 7, 21)
    else:
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        out = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return out.astype(np.float32) / 255.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)))


def _artifact_safe_blend(
    observed: np.ndarray,
    initial: np.ndarray,
    candidate: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    """Reject or attenuate refinements that buy residual gain with visible artifacts."""
    y = np.asarray(observed, dtype=np.float32)
    base = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0)
    cand = np.clip(np.asarray(candidate, dtype=np.float32), 0.0, 1.0)
    kernel_size = int(max(kernel.shape))

    base_diag = artifact_diagnostics(y, base)
    cand_diag = artifact_diagnostics(y, cand)
    base_risky = ripple_risk(base_diag, kernel_size=kernel_size)
    cand_risky = ripple_risk(cand_diag, kernel_size=kernel_size)
    if cand_risky:
        return base.copy()
    if base_risky and (
        cand_diag.noise_ratio >= base_diag.noise_ratio * 0.95
        or cand_diag.highpass_ratio >= base_diag.highpass_ratio * 0.95
    ):
        return base.copy()

    base_reblur = reblur_image(base, kernel, workers=workers)
    cand_reblur = reblur_image(cand, kernel, workers=workers)
    base_rmse = _rmse(base_reblur, y)
    cand_rmse = _rmse(cand_reblur, y)
    if not np.isfinite(cand_rmse) or cand_rmse >= base_rmse * 0.997:
        return base.copy()

    base_score, _ = restoration_score(y, base, base_reblur)
    cand_score, _ = restoration_score(y, cand, cand_reblur)
    if cand_score >= base_score * 0.995:
        return base.copy()

    relative_gain = (base_rmse - cand_rmse) / max(base_rmse, 1e-8)
    noise_ratio = noise_mad(cand) / max(noise_mad(base), 5e-4)
    highpass_ratio = highpass_rms(cand) / max(highpass_rms(base), 2e-3)
    edge_ratio = edge_energy(cand) / max(edge_energy(base), 0.02)
    clip_growth = max(0.0, clipping_fraction(cand) - clipping_fraction(base))

    # Larger fidelity gains are allowed somewhat more detail, but never the 10x-30x
    # high-frequency explosions seen in earlier benchmark runs.
    allowed_noise = 1.20 + 0.65 * min(relative_gain / 0.30, 1.0)
    allowed_highpass = 1.20 + 0.55 * min(relative_gain / 0.30, 1.0)
    if noise_ratio > allowed_noise * 1.35:
        return base.copy()
    if highpass_ratio > allowed_highpass * 1.35:
        return base.copy()
    if edge_ratio > 1.55 or clip_growth > 0.025:
        return base.copy()

    fidelity_alpha = float(np.clip(relative_gain / 0.18, 0.10, 1.0))
    noise_alpha = min(1.0, allowed_noise / max(noise_ratio, 1e-8))
    highpass_alpha = min(1.0, allowed_highpass / max(highpass_ratio, 1e-8))
    alpha = float(np.clip(fidelity_alpha * noise_alpha * highpass_alpha, 0.0, 1.0))

    blended = np.clip(base + alpha * (cand - base), 0.0, 1.0).astype(np.float32)
    if ripple_risk(
        artifact_diagnostics(y, blended),
        kernel_size=kernel_size,
    ):
        return base.copy()
    blended_reblur = reblur_image(blended, kernel, workers=workers)
    blended_score, _ = restoration_score(y, blended, blended_reblur)
    if blended_score >= base_score:
        return base.copy()
    return blended


def annealed_pnp_refine(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    steps: int = 4,
    sigma_start: float = 0.025,
    sigma_end: float = 0.004,
    candidates: int = 2,
    seed: int = 0,
    workers: int = -1,
) -> np.ndarray:
    """Annealed stochastic plug-and-play refinement with artifact protection."""
    if steps < 1 or candidates < 1:
        raise ValueError("steps and candidates must be >= 1")
    if sigma_start <= 0 or sigma_end <= 0:
        raise ValueError("sigma values must be > 0")

    y = np.asarray(observed, dtype=np.float32)
    initial_arr = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0)
    x = initial_arr.copy()
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(float(sigma_start), float(sigma_end), int(steps))

    for index, sigma in enumerate(sigmas):
        rho = 0.05 + 0.10 * index / max(steps - 1, 1)
        best_score = math.inf
        best = x
        for _ in range(candidates):
            noise = rng.normal(0.0, sigma, x.shape).astype(np.float32)
            noisy = np.clip(x + noise, 0.0, 1.0)
            prior = _nlm_gaussian_denoiser(noisy, float(sigma))
            candidate = _data_consistency(y, prior, kernel, rho, workers=workers)
            reblurred = reblur_image(candidate, kernel, workers=workers)
            score, _ = restoration_score(y, candidate, reblurred)
            if score < best_score:
                best_score = score
                best = candidate
        x = best

    return _artifact_safe_blend(y, initial_arr, x, kernel, workers=workers)


def extreme_channel_refine(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    steps: int = 3,
    patch_size: int = 15,
    workers: int = -1,
) -> np.ndarray:
    """Dual-extreme local-contrast refinement with artifact protection."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")

    y = np.asarray(observed, dtype=np.float32)
    initial_arr = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0)
    x = initial_arr.copy()
    if x.ndim != 3 or x.shape[2] != 3:
        return x

    footprint = np.ones((patch_size, patch_size), dtype=np.uint8)
    for index in range(steps):
        dark = cv2.erode(np.min(x, axis=2), footprint)
        bright = cv2.dilate(np.max(x, axis=2), footprint)
        dark_weight = np.clip((0.10 - dark) / 0.10, 0.0, 1.0)
        bright_weight = np.clip((bright - 0.90) / 0.10, 0.0, 1.0)
        extreme_weight = np.maximum(dark_weight, bright_weight)[..., None]

        smooth = cv2.bilateralFilter(x.astype(np.float32), d=0, sigmaColor=0.06, sigmaSpace=2.0)
        detail = x - smooth
        gain = 0.04 + 0.08 * extreme_weight
        prior = np.clip(x + gain * detail, 0.0, 1.0)
        prior *= 1.0 - 0.010 * dark_weight[..., None]
        prior = 1.0 - (1.0 - prior) * (1.0 - 0.010 * bright_weight[..., None])
        rho = 0.12 + 0.06 * index
        x = _data_consistency(y, prior, kernel, rho, workers=workers)

    return _artifact_safe_blend(y, initial_arr, x, kernel, workers=workers)
