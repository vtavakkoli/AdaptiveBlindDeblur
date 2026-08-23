from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import fft

from .fft_utils import psf2otf


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
    """Apply the estimated PSF with reflection padding for quality diagnostics."""
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
    """Closed-form proximal step for ||Kx-y||^2 + rho ||x-z||^2."""
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
    """Fast weight-free Gaussian denoiser used as a PnP prior."""
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


def _noise_mad(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    median = float(np.median(lap))
    return float(np.median(np.abs(lap - median)))


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
    """Diffusion-inspired annealed plug-and-play refinement.

    This is intentionally *not* marketed as a learned diffusion model. It borrows
    the useful restoration structure of modern diffusion/PnP methods: an
    annealed Gaussian-noise schedule, a denoising prior, and an explicit
    measurement-consistency proximal step. The denoiser is OpenCV NLM so the
    method stays weight-free, deterministic, CPU-friendly, and Docker-ready.
    """
    if steps < 1 or candidates < 1:
        raise ValueError("steps and candidates must be >= 1")
    if sigma_start <= 0 or sigma_end <= 0:
        raise ValueError("sigma values must be > 0")

    y = np.asarray(observed, dtype=np.float32)
    x = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0).copy()
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(float(sigma_start), float(sigma_end), int(steps))

    for index, sigma in enumerate(sigmas):
        rho = 0.04 + 0.08 * index / max(steps - 1, 1)
        best_score = math.inf
        best = x
        for _ in range(candidates):
            noise = rng.normal(0.0, sigma, x.shape).astype(np.float32)
            noisy = np.clip(x + noise, 0.0, 1.0)
            prior = _nlm_gaussian_denoiser(noisy, float(sigma))
            candidate = _data_consistency(y, prior, kernel, rho, workers=workers)
            residual = float(np.sqrt(np.mean((reblur_image(candidate, kernel, workers=workers) - y) ** 2)))
            # The small MAD term rejects noisy candidates without explicitly
            # rewarding over-sharpening.
            score = residual + 0.025 * _noise_mad(candidate)
            if score < best_score:
                best_score = score
                best = candidate
        x = best
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def extreme_channel_refine(
    observed: np.ndarray,
    initial: np.ndarray,
    kernel: np.ndarray,
    *,
    steps: int = 3,
    patch_size: int = 15,
    workers: int = -1,
) -> np.ndarray:
    """Extreme-channel guided refinement using both dark and bright evidence.

    Yan et al.'s Extreme Channels Prior showed why dark-channel-only blind
    deblurring can be weak on bright/saturated scenes. This lightweight
    refinement does not claim to reproduce their full optimizer; it uses the
    same core signal -- local dark and bright extrema -- to gate detail
    recovery, then projects every step back to the observed blur model.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")

    y = np.asarray(observed, dtype=np.float32)
    x = np.clip(np.asarray(initial, dtype=np.float32), 0.0, 1.0).copy()
    if x.ndim != 3 or x.shape[2] != 3:
        # The local bright/dark RGB extrema are the point of this refinement.
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
        gain = 0.12 + 0.20 * extreme_weight
        prior = np.clip(x + gain * detail, 0.0, 1.0)

        # Gently strengthen trustworthy local extrema instead of globally
        # stretching contrast, which would make saturated scenes unstable.
        prior *= 1.0 - 0.025 * dark_weight[..., None]
        prior = 1.0 - (1.0 - prior) * (1.0 - 0.025 * bright_weight[..., None])
        rho = 0.08 + 0.04 * index
        x = _data_consistency(y, prior, kernel, rho, workers=workers)

    return np.clip(x, 0.0, 1.0).astype(np.float32)
