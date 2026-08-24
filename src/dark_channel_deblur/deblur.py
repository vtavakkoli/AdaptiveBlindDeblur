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
from .quality import (
    artifact_diagnostics,
    artifact_severity,
    restoration_score,
    ripple_risk,
    saturation_checkpoint_safe,
    saturation_instability,
    should_retry_kernel,
)
from .refinement import reblur_image
from .saturation import whyte_deconvolution


def _matlab_rgb2gray(image: np.ndarray) -> np.ndarray:
    """Match the coefficients used by MATLAB rgb2gray for floating RGB data."""
    arr = np.asarray(image, dtype=np.float32)
    return (
        arr[..., 0] * np.float32(0.298936021293775)
        + arr[..., 1] * np.float32(0.587043074451121)
        + arr[..., 2] * np.float32(0.114020904255103)
    ).astype(np.float32)


def _downsample(image: np.ndarray, ratio: float) -> np.ndarray:
    """Faithful translation of the release's Levin ``downSmpImC`` routine."""
    arr = np.asarray(image, dtype=np.float32)
    if ratio == 1.0:
        return arr.copy()

    sigma = ratio / math.pi
    grid = np.arange(-50, 51, dtype=np.float64) * (2.0 * math.pi)
    kernel = np.exp(-0.5 * grid * grid * sigma * sigma)
    kernel /= kernel.sum()
    cumulative = np.cumsum(kernel)
    cumulative = np.minimum(cumulative, cumulative[::-1])
    kernel = kernel[cumulative > 0.05].astype(np.float32)

    # The MATLAB code uses conv2(...,'valid') before bilinear interp2 sampling.
    filtered = cv2.sepFilter2D(
        arr,
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    radius = kernel.size // 2
    if radius:
        filtered = filtered[radius:-radius, radius:-radius]

    step = 1.0 / ratio
    xs = np.arange(0.0, filtered.shape[1], step, dtype=np.float32)
    ys = np.arange(0.0, filtered.shape[0], step, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return cv2.remap(
        filtered,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float32)


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
        y = cv2.resize(y, None, fx=cfg.prescale, fy=cfg.prescale, interpolation=cv2.INTER_CUBIC)
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


def _prefer_candidate(
    current_score: float,
    current_diag: object,
    candidate_score: float,
    candidate_diag: object,
    *,
    kernel_size: int,
) -> bool:
    """Prefer a safe candidate over a ripple-risky one before comparing residual score."""
    current_risky = ripple_risk(current_diag, kernel_size=kernel_size)
    candidate_risky = ripple_risk(candidate_diag, kernel_size=kernel_size)
    if current_risky and not candidate_risky:
        return candidate_score <= current_score * 1.35
    if not current_risky and candidate_risky:
        return False
    return candidate_score < current_score


def _restore_saturated(
    image: np.ndarray,
    kernel: np.ndarray,
    config: DeblurConfig,
) -> tuple[np.ndarray, float, str]:
    """Run Whyte RL, with adaptive early stopping only when late iterations destabilize."""
    full_iterations = int(config.saturation_iterations)
    restored = whyte_deconvolution(
        image,
        kernel,
        iterations=full_iterations,
        workers=config.fft_workers,
    )
    score, _ = _candidate_score(image, kernel, restored, config)
    full_diag = artifact_diagnostics(image, restored)

    if not config.conservative_restoration or not saturation_instability(full_diag):
        return restored.astype(np.float32), float(score), f"whyte_{full_iterations}"

    # Whyte RL is excellent for saturated highlights, but late iterations can amplify
    # night-scene sensor noise.  Probe earlier checkpoints only after instability is
    # detected, then keep the latest checkpoint still inside conservative budgets.
    trial_iterations = sorted(
        {
            max(5, int(round(full_iterations * 0.30))),
            max(8, int(round(full_iterations * 0.40))),
            max(10, int(round(full_iterations * 0.60))),
        }
    )
    trial_iterations = [value for value in trial_iterations if value < full_iterations]

    candidates: list[tuple[int, np.ndarray, object, float]] = []
    for iterations in trial_iterations:
        candidate = whyte_deconvolution(
            image,
            kernel,
            iterations=iterations,
            workers=config.fft_workers,
        )
        candidate_diag = artifact_diagnostics(image, candidate)
        candidate_score, _ = _candidate_score(image, kernel, candidate, config)
        candidates.append((iterations, candidate, candidate_diag, float(candidate_score)))

    safe = [item for item in candidates if saturation_checkpoint_safe(item[2])]
    if safe:
        iterations, candidate, candidate_diag, candidate_score = max(safe, key=lambda item: item[0])
    elif candidates:
        iterations, candidate, candidate_diag, candidate_score = min(
            candidates,
            key=lambda item: artifact_severity(item[2]),
        )
    else:
        return restored.astype(np.float32), float(score), f"whyte_{full_iterations}"

    if artifact_severity(candidate_diag) < artifact_severity(full_diag) * 0.85:
        return candidate.astype(np.float32), candidate_score, f"whyte_guarded_{iterations}"
    return restored.astype(np.float32), float(score), f"whyte_{full_iterations}"


def _restore_and_score(
    image: np.ndarray,
    kernel: np.ndarray,
    config: DeblurConfig,
) -> tuple[np.ndarray, float, str]:
    """Run final restoration with adaptive guards against ripple and over-iteration."""
    if config.saturated:
        return _restore_saturated(image, kernel, config)

    configured = ringing_artifacts_removal(image, kernel, config)
    best_image = configured
    best_score, diag = _candidate_score(image, kernel, configured, config)
    best_name = "configured"

    suspicious = (
        best_score > 0.03
        or diag.clipping_growth > 0.04
        or diag.noise_ratio > 1.8
        or ripple_risk(diag, kernel_size=config.kernel_size)
    )
    if not config.conservative_restoration or not suspicious:
        return best_image.astype(np.float32), float(best_score), best_name

    conservative_cfg = replace(
        config,
        lambda_tv=max(config.lambda_tv * 2.5, 1e-3),
        lambda_l0=max(config.lambda_l0 * 1.5, 7.5e-4),
        weight_ring=max(config.weight_ring, 0.65),
    )
    conservative = ringing_artifacts_removal(image, kernel, conservative_cfg)
    conservative_score, conservative_diag = _candidate_score(
        image, kernel, conservative, config
    )
    if _prefer_candidate(
        float(best_score),
        diag,
        float(conservative_score),
        conservative_diag,
        kernel_size=config.kernel_size,
    ):
        best_image = conservative
        best_score = conservative_score
        best_name = "conservative"
        diag = conservative_diag

    still_suspicious = (
        best_score > 0.045
        or diag.clipping_growth > 0.06
        or ripple_risk(diag, kernel_size=config.kernel_size)
    )
    if still_suspicious:
        tv_safe_cfg = replace(
            config,
            lambda_tv=max(config.lambda_tv * 5.0, 2e-3),
            lambda_l0=max(config.lambda_l0 * 2.0, 1e-3),
            weight_ring=max(config.weight_ring, 0.85),
        )
        tv_safe = ringing_artifacts_removal(image, kernel, tv_safe_cfg)
        tv_score, tv_diag = _candidate_score(image, kernel, tv_safe, config)
        if _prefer_candidate(
            float(best_score),
            diag,
            float(tv_score),
            tv_diag,
            kernel_size=config.kernel_size,
        ):
            best_image = tv_safe
            best_score = tv_score
            best_name = "tv_safe"
            diag = tv_diag

    # Last-resort guard: if all deconvolution candidates still show the characteristic
    # long-kernel ripple pattern, attenuate only the deconvolution delta.  This is
    # preferable to returning a visually unstable image and never uses legacy pixels.
    if ripple_risk(diag, kernel_size=config.kernel_size):
        observed = np.asarray(image, dtype=np.float32)
        for alpha in (0.85, 0.70, 0.55, 0.40):
            blended = np.clip(observed + alpha * (best_image - observed), 0.0, 1.0)
            blend_score, blend_diag = _candidate_score(image, kernel, blended, config)
            if not ripple_risk(blend_diag, kernel_size=config.kernel_size):
                best_image = blended
                best_score = blend_score
                best_name = f"ripple_guard_{alpha:.2f}"
                break

    return best_image.astype(np.float32), float(best_score), best_name


def deblur_image(
    image: np.ndarray,
    config: DeblurConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blind-deblur an RGB/gray float image and return result, kernel, interim latent."""
    cfg = config or DeblurConfig()
    cfg.validate()
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        gray = _matlab_rgb2gray(arr)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError("image must be HxW or HxWx3")

    kernel, interim = estimate_blur_kernel(gray, cfg)
    result, primary_score, _ = _restore_and_score(arr, kernel, cfg)

    if (
        cfg.robust_selection
        and cfg.retry_gradient_only
        and not cfg.saturated
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
        primary_diag = artifact_diagnostics(arr, result)
        retry_diag = artifact_diagnostics(arr, retry_result)
        primary_risky = ripple_risk(primary_diag, kernel_size=cfg.kernel_size)
        retry_risky = ripple_risk(retry_diag, kernel_size=cfg.kernel_size)
        choose_retry = (
            (primary_risky and not retry_risky and retry_score <= primary_score * 1.35)
            or (not retry_risky and retry_score < primary_score * 0.97)
        )
        if choose_retry:
            result = retry_result
            kernel = retry_kernel
            interim = retry_interim

    return result.astype(np.float32), kernel.astype(np.float32), interim.astype(np.float32)
