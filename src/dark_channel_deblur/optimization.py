from __future__ import annotations

import cv2
import numpy as np
from scipy import fft

from .config import DeblurConfig
from .dark_channel import project_dark_channel
from .fft_utils import fast_shape, psf2otf
from .boundary import wrap_boundary


def _periodic_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.roll(image, -1, axis=1) - image
    gy = np.roll(image, -1, axis=0) - image
    return gx, gy


def _divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return (np.roll(gx, 1, axis=1) - gx) + (np.roll(gy, 1, axis=0) - gy)


def _gradient_denominator(shape: tuple[int, int], workers: int) -> np.ndarray:
    fx = psf2otf(np.array([[1.0, -1.0]], dtype=np.float32), shape, workers)
    fy = psf2otf(np.array([[1.0], [-1.0]], dtype=np.float32), shape, workers)
    return (np.abs(fx) ** 2 + np.abs(fy) ** 2).astype(np.float32)


def _otsu_threshold01(values: np.ndarray) -> float:
    clipped = np.clip(values, 0.0, 1.0)
    u8 = np.rint(clipped * 255.0).astype(np.uint8)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold) / 255.0, 1.0 / 255.0)


def l0_deblur_dark_channel(
    blurred: np.ndarray,
    kernel: np.ndarray,
    lambda_dark: float,
    lambda_grad: float,
    config: DeblurConfig,
) -> np.ndarray:
    """Solve the latent-image subproblem with dark-channel + L0 gradients."""
    image = np.asarray(blurred, dtype=np.float32)
    s = image.copy()
    shape = s.shape[:2]
    workers = config.fft_workers

    ker = psf2otf(kernel, shape, workers)
    den_kernel = (np.abs(ker) ** 2).astype(np.float32)
    den_grad = _gradient_denominator(shape, workers)
    if s.ndim == 3:
        ker_c = ker[..., None]
        den_kernel_c = den_kernel[..., None]
        den_grad_c = den_grad[..., None]
    else:
        ker_c = ker
        den_kernel_c = den_kernel
        den_grad_c = den_grad

    norm1 = np.conj(ker_c) * fft.fft2(s, axes=(0, 1), workers=workers)

    beta_pixel = lambda_dark / _otsu_threshold01(s * s)
    dark_steps = 0
    while beta_pixel < config.beta_max_pixel:
        if config.max_dark_steps is not None and dark_steps >= config.max_dark_steps:
            break
        u = project_dark_channel(s, lambda_dark, beta_pixel, config.dark_patch_size)

        beta = max(2.0 * lambda_grad, 1e-8)
        grad_steps = 0
        while beta < config.beta_max_grad:
            if config.max_grad_steps is not None and grad_steps >= config.max_grad_steps:
                break
            gx, gy = _periodic_gradients(s)
            energy = gx * gx + gy * gy
            if s.ndim == 3:
                keep_zero = np.sum(energy, axis=2) < (lambda_grad / beta)
                gx[keep_zero, :] = 0.0
                gy[keep_zero, :] = 0.0
            else:
                keep_zero = energy < (lambda_grad / beta)
                gx[keep_zero] = 0.0
                gy[keep_zero] = 0.0

            div = _divergence(gx, gy)
            denominator = den_kernel_c + beta * den_grad_c + beta_pixel
            numerator = (
                norm1
                + beta * fft.fft2(div, axes=(0, 1), workers=workers)
                + beta_pixel * fft.fft2(u, axes=(0, 1), workers=workers)
            )
            s = fft.ifft2(numerator / denominator, axes=(0, 1), workers=workers).real.astype(np.float32)
            beta *= config.kappa
            grad_steps += 1
            if lambda_grad == 0:
                break

        beta_pixel *= config.kappa
        dark_steps += 1
    return s


def l0_restoration(
    blurred: np.ndarray,
    kernel: np.ndarray,
    lambda_l0: float,
    config: DeblurConfig,
    *,
    pad: bool = True,
) -> np.ndarray:
    """L0 gradient restoration from Xu et al./Pan et al."""
    image = np.asarray(blurred, dtype=np.float32)
    original_shape = image.shape[:2]
    if pad:
        target = fast_shape(original_shape, kernel.shape)
        image = wrap_boundary(image, target)
    s = image.copy()
    shape = s.shape[:2]
    workers = config.fft_workers

    ker = psf2otf(kernel, shape, workers)
    den_kernel = np.abs(ker) ** 2
    den_grad = _gradient_denominator(shape, workers)
    if s.ndim == 3:
        ker = ker[..., None]
        den_kernel = den_kernel[..., None]
        den_grad = den_grad[..., None]
    norm1 = np.conj(ker) * fft.fft2(s, axes=(0, 1), workers=workers)

    beta = max(2.0 * lambda_l0, 1e-8)
    steps = 0
    while beta < config.beta_max_grad:
        if config.max_grad_steps is not None and steps >= config.max_grad_steps:
            break
        gx, gy = _periodic_gradients(s)
        energy = gx * gx + gy * gy
        if s.ndim == 3:
            mask = np.sum(energy, axis=2) < (lambda_l0 / beta)
            gx[mask, :] = 0.0
            gy[mask, :] = 0.0
        else:
            mask = energy < (lambda_l0 / beta)
            gx[mask] = 0.0
            gy[mask] = 0.0
        div = _divergence(gx, gy)
        numerator = norm1 + beta * fft.fft2(div, axes=(0, 1), workers=workers)
        s = fft.ifft2(numerator / (den_kernel + beta * den_grad), axes=(0, 1), workers=workers).real.astype(np.float32)
        beta *= config.kappa
        steps += 1

    return s[: original_shape[0], : original_shape[1], ...]


def tv_deconvolution_aniso(
    blurred: np.ndarray,
    kernel: np.ndarray,
    lambda_tv: float,
    config: DeblurConfig,
) -> np.ndarray:
    """Fast anisotropic TV-L2 deconvolution (ADM/Split-Bregman style)."""
    image = np.asarray(blurred, dtype=np.float32)
    shape = image.shape[:2]
    workers = config.fft_workers
    ker = psf2otf(kernel, shape, workers)
    den1 = np.abs(ker) ** 2
    den2 = _gradient_denominator(shape, workers)
    if image.ndim == 3:
        ker = ker[..., None]
        den1 = den1[..., None]
        den2 = den2[..., None]
    norm1 = np.conj(ker) * fft.fft2(image, axes=(0, 1), workers=workers)

    result = image.copy()
    gx, gy = _periodic_gradients(result)
    beta = 1.0 / max(lambda_tv, 1e-12)
    steps = 0
    while beta > 1e-3:
        if config.max_grad_steps is not None and steps >= config.max_grad_steps:
            break
        gamma = 1.0 / (2.0 * beta)
        wx = np.sign(gx) * np.maximum(np.abs(gx) - beta * lambda_tv, 0.0)
        wy = np.sign(gy) * np.maximum(np.abs(gy) - beta * lambda_tv, 0.0)
        div = _divergence(wx, wy)
        result = fft.ifft2(
            (norm1 + gamma * fft.fft2(div, axes=(0, 1), workers=workers)) / (den1 + gamma * den2),
            axes=(0, 1),
            workers=workers,
        ).real.astype(np.float32)
        gx, gy = _periodic_gradients(result)
        beta /= 2.0
        steps += 1
    return result


def ringing_artifacts_removal(
    image: np.ndarray,
    kernel: np.ndarray,
    config: DeblurConfig,
) -> np.ndarray:
    """TV + L0 final restoration with bilateral ringing suppression."""
    original_shape = image.shape[:2]
    target = fast_shape(original_shape, kernel.shape)
    padded = wrap_boundary(np.asarray(image, dtype=np.float32), target)

    latent_tv = tv_deconvolution_aniso(padded, kernel, config.lambda_tv, config)
    latent_tv = latent_tv[: original_shape[0], : original_shape[1], ...]
    if config.weight_ring == 0:
        return np.clip(latent_tv, 0.0, 1.0)

    latent_l0 = l0_restoration(padded, kernel, config.lambda_l0, config, pad=False)
    latent_l0 = latent_l0[: original_shape[0], : original_shape[1], ...]
    diff = (latent_tv - latent_l0).astype(np.float32)
    filtered = cv2.bilateralFilter(diff, d=0, sigmaColor=0.1, sigmaSpace=3.0)
    return np.clip(latent_tv - config.weight_ring * filtered, 0.0, 1.0)
