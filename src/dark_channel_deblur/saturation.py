from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import fft

from .fft_utils import psf2otf


def _gaussian_kernel(size: int = 21, sigma: float = 3.0) -> np.ndarray:
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    one_d = np.exp(-(axis * axis) / (2.0 * sigma * sigma))
    one_d /= one_d.sum()
    return np.outer(one_d, one_d)


def whyte_deconvolution(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 50,
    saturation_threshold: float = 1.0,
    saturation_smoothness: float = 50.0,
    workers: int = -1,
) -> np.ndarray:
    """Deblur partially saturated images using Whyte et al.'s guarded RL update.

    This is the uniform-blur branch used by ``whyte_deconv.m`` / ``deconvRL.m``
    in the MATLAB release. The input is gamma-linearized, optimized with the
    smooth saturation forward model and ringing-prevention masks, then converted
    back to display gamma.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    observed = np.asarray(image, dtype=np.float64)
    if observed.ndim == 2:
        observed = observed[..., None]
        squeeze = True
    elif observed.ndim == 3:
        squeeze = False
    else:
        raise ValueError("image must be HxW or HxWxC")

    observed_linear = np.maximum(observed, 0.0) ** 2.2
    psf = np.asarray(kernel, dtype=np.float64).copy()
    peak = float(psf.max())
    if peak <= 0:
        raise ValueError("kernel must contain positive mass")
    psf[psf < peak / 100.0] = 0.0
    psf /= psf.sum()

    kh, kw = psf.shape
    top = math.ceil((kh - 1) / 2)
    bottom = math.floor((kh - 1) / 2)
    left = math.ceil((kw - 1) / 2)
    right = math.floor((kw - 1) / 2)
    padded = np.pad(
        observed_linear,
        ((top, bottom), (left, right), (0, 0)),
        mode="edge",
    )
    h, w, channels = padded.shape

    kernel_fft = psf2otf(psf, (h, w), workers)
    support_fft = psf2otf((psf != 0).astype(np.float64), (h, w), workers)

    mask = np.zeros_like(padded, dtype=np.float64)
    y_stop = h - bottom if bottom else h
    x_stop = w - right if right else w
    mask[top:y_stop, left:x_stop, :] = 1.0

    estimate = padded.copy()
    yy, xx = np.mgrid[-3:4, -3:4]
    dilation_disk = ((xx * xx + yy * yy) <= 9).astype(np.uint8)
    smooth_filter = _gaussian_kernel(21, 3.0)

    def blur(value: np.ndarray) -> np.ndarray:
        spectrum = fft.fft2(value, axes=(0, 1), workers=workers)
        return fft.ifft2(
            spectrum * kernel_fft[..., None], axes=(0, 1), workers=workers
        ).real

    def conjugate_blur(value: np.ndarray) -> np.ndarray:
        spectrum = fft.fft2(value, axes=(0, 1), workers=workers)
        return fft.ifft2(
            spectrum * np.conj(kernel_fft)[..., None], axes=(0, 1), workers=workers
        ).real

    def dilate_influence(value: np.ndarray) -> np.ndarray:
        spectrum = fft.fft2(value, axes=(0, 1), workers=workers)
        influenced = fft.ifft2(
            spectrum * support_fft[..., None], axes=(0, 1), workers=workers
        ).real
        return np.minimum(influenced, 1.0)

    for _ in range(iterations):
        linear_prediction = np.maximum(blur(estimate), 0.0)
        z = saturation_smoothness * (linear_prediction - saturation_threshold)
        saturated_prediction = linear_prediction - np.logaddexp(0.0, z) / saturation_smoothness
        saturation_gradient = 1.0 / (1.0 + np.exp(np.clip(z, -700.0, 700.0)))

        error_ratio = padded / np.maximum(saturated_prediction, np.finfo(np.float64).eps)
        masked_error = (error_ratio - 1.0) * mask * saturation_gradient

        hard_mask = np.empty_like(estimate)
        for channel in range(channels):
            hard_mask[..., channel] = cv2.dilate(
                (estimate[..., channel] >= 0.9).astype(np.uint8),
                dilation_disk,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        valid_mask = 1.0 - dilate_influence(hard_mask)

        update_u = conjugate_blur(masked_error * valid_mask) + 1.0
        update_s = conjugate_blur(masked_error) + 1.0
        weights = np.empty_like(estimate)
        for channel in range(channels):
            weights[..., channel] = cv2.filter2D(
                hard_mask[..., channel],
                -1,
                smooth_filter,
                borderType=cv2.BORDER_CONSTANT,
            )
        update = np.maximum(update_u + (update_s - update_u) * weights, 0.0)
        estimate *= update

    estimate = estimate[top:y_stop, left:x_stop, :]
    result = np.maximum(estimate, 0.0) ** (1.0 / 2.2)
    result = result.astype(np.float32)
    return result[..., 0] if squeeze else result
