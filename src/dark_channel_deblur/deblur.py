from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from .boundary import wrap_boundary
from .config import DeblurConfig
from .fft_utils import fast_shape
from .kernel import (
    adjust_psf_center,
    estimate_psf,
    init_kernel,
    prune_kernel,
    resize_kernel,
    threshold_gradients,
    valid_gradients,
)
from .optimization import l0_deblur_dark_channel, l0_restoration, ringing_artifacts_removal
from .quality import restoration_score, should_retry_kernel
from .refinement import reblur_image


def _downsample(image: np.ndarray, ratio: float) -> np.ndarray:
    if ratio == 1.0:
        return image.copy()
    sigma = max(0.01, 1.0 / math.pi * ratio)
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    h = max(2, int(round(image.shape[0] * ratio)))
    w = max(2, int(round(image.shape[1] * ratio)))
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_AREA)


def estimate_blur_kernel(
    gray: np.ndarray,
    config: DeblurConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a motion blur kernel from a grayscale image in [0, 1]."""
    cfg = config or DeblurConfig()
    cfg.validate()
    y = np.asarray(gray, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("estimate_blur_kernel expects a 2-D grayscale image")
    if cfg.prescale != 1.0:
        y = cv2.resize(y, None, fx=cfg.prescale, fy=cfg.prescale, interpolation=cv2.INTER_AREA)
    if cfg.gamma_correct != 1.0:
        y = np.power(np.clip(y, 0.0, 1.0), cfg.gamma_correct, dtype=np.float32)

    ratio = math.sqrt(0.5)
    max_iter = max(int(math.floor(math.log(5.0 / cfg.kernel_size) / math.log(ratio))), 0)
    scale_values = ratio ** np.arange(max_iter + 1, dtype=np.float64)
    kernel_sizes = np.ceil(cfg.kernel_size * scale_values).astype(int)
    kernel_sizes += (kernel_sizes % 2 == 0).astype(int)

    threshold: float | None = None
    kernel: np.ndarray | None = None
    latent = y.copy()
    lambda_dark = cfg.lambda_dark
    lambda_grad = cfg.lambda_grad

    for scale_idx in range(max_iter, -1, -1):
        size = int(kernel_sizes[scale_idx])
        if kernel is None:
            kernel = init_kernel(size)
        else:
            kernel = resize_kernel(kernel, 1.0 / ratio, size)
        ys = _downsample(y, float(scale_values[scale_idx]))

        target = fast_shape(ys.shape, kernel.shape)
        padded = wrap_boundary(ys, target)
        bx, by = valid_gradients(padded[: ys.shape[0], : ys.shape[1]])

        if threshold is None:
            _, _, threshold = threshold_gradients(ys, size, None)

        for _ in range(cfg.xk_iter):
            if lambda_dark != 0:
                latent_padded = l0_deblur_dark_channel(padded, kernel, lambda_dark, lambda_grad, cfg)
                latent = latent_padded[: ys.shape[0], : ys.shape[1]]
            else:
                latent = l0_restoration(ys, kernel, lambda_grad, cfg)

            lx, ly, threshold = threshold_gradients(latent, size, threshold)
            kernel = estimate_psf(
                bx,
                by,
                lx,
                ly,
                weight=2.0,
                psf_shape=kernel.shape,
                workers=cfg.fft_workers,
            )
            kernel = prune_kernel(kernel)
            lambda_dark = max(lambda_dark / 1.1, 1e-4) if lambda_dark else 0.0
            lambda_grad = max(lambda_grad / 1.1, 1e-4) if lambda_grad else 0.0

        kernel = adjust_psf_center(kernel)

    assert kernel is not None
    if cfg.k_thresh > 0 and np.max(kernel) > 0:
        kernel[kernel < np.max(kernel) / cfg.k_thresh] = 0.0
    kernel = np.maximum(kernel, 0.0)
    total = float(kernel.sum())
    if total <= 0:
        kernel = init_kernel(cfg.kernel_size)
    else:
        kernel /= total
    return kernel.astype(np.float32), np.clip(latent, 0.0, 1.0).astype(np.float32)


def _candidate_score(
    image: np.ndarray,
    kernel: np.ndarray,
    restored: np.ndarray,
    config: DeblurConfig,
) -> tuple[float, object]:
    reblurred = reblur_image(restored, kernel, workers=config.fft_workers)
    return restoration_score(image, restored, reblurred)


def _restore_and_score(
    image: np.ndarray,
    kernel: np.ndarray,
    config: DeblurConfig,
) -> tuple[np.ndarray, float, str]:
    """Run extra restoration candidates only when the configured one is suspicious."""
    configured = ringing_artifacts_removal(image, kernel, config)
    best_image = configured
    best_score, diag = _candidate_score(image, kernel, configured, config)
    best_name = "configured"

    suspicious = (
        best_score > 0.03
        or diag.clipping_growth > 0.04
        or (diag.edge_ratio > 2.8 and diag.highpass_ratio > 4.0)
        or diag.noise_ratio > 1.8
    )
    if not config.conservative_restoration or not suspicious:
        return best_image.astype(np.float32), float(best_score), best_name

    conservative_cfg = replace(
        config,
        lambda_tv=max(config.lambda_tv * 2.0, 1e-3),
        lambda_l0=max(config.lambda_l0 * 2.0, 1e-3),
        weight_ring=min(config.weight_ring, 0.5),
    )
    conservative = ringing_artifacts_removal(image, kernel, conservative_cfg)
    conservative_score, conservative_diag = _candidate_score(
        image, kernel, conservative, config
    )
    if conservative_score < best_score:
        best_image = conservative
        best_score = conservative_score
        best_name = "conservative"
        diag = conservative_diag

    still_suspicious = (
        best_score > 0.045
        or diag.clipping_growth > 0.06
        or (diag.edge_ratio > 3.0 and diag.highpass_ratio > 4.5)
    )
    if still_suspicious:
        tv_safe_cfg = replace(
            config,
            lambda_tv=max(config.lambda_tv * 3.0, 2e-3),
            weight_ring=0.0,
        )
        tv_safe = ringing_artifacts_removal(image, kernel, tv_safe_cfg)
        tv_score, _ = _candidate_score(image, kernel, tv_safe, config)
        if tv_score < best_score:
            best_image = tv_safe
            best_score = tv_score
            best_name = "tv_safe"

    return best_image.astype(np.float32), float(best_score), best_name


def deblur_image(
    image: np.ndarray,
    config: DeblurConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blind-deblur an RGB/gray float image and return result, kernel, interim latent.

    Robust mode first estimates the configured PSF, conditionally selects a more
    conservative final restoration, and retries suspicious PSFs with a gradient-only
    estimator. Legacy/reference pixels are never used by inference or selection.
    """
    cfg = config or DeblurConfig()
    cfg.validate()
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError("image must be HxW or HxWx3")

    kernel, interim = estimate_blur_kernel(gray, cfg)
    result, primary_score, _ = _restore_and_score(arr, kernel, cfg)

    if (
        cfg.robust_selection
        and cfg.retry_gradient_only
        and cfg.lambda_dark != 0
        and should_retry_kernel(
            arr,
            result,
            kernel,
            kernel_size=cfg.kernel_size,
            blind_score=primary_score,
        )
    ):
        retry_cfg = replace(
            cfg,
            lambda_dark=0.0,
            gamma_correct=1.0,
            retry_gradient_only=False,
        )
        retry_kernel, retry_interim = estimate_blur_kernel(gray, retry_cfg)
        retry_result, retry_score, _ = _restore_and_score(arr, retry_kernel, retry_cfg)
        if retry_score < primary_score * 0.97:
            result = retry_result
            kernel = retry_kernel
            interim = retry_interim

    return result.astype(np.float32), kernel.astype(np.float32), interim.astype(np.float32)
