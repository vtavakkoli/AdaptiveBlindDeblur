from __future__ import annotations

import cv2
import numpy as np

from dark_channel_deblur.config import DeblurConfig
from dark_channel_deblur.deblur import deblur_image, estimate_blur_kernel


def _synthetic_image(size: int = 48) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    image[8:38, 10:14] = 1.0
    image[12:16, 8:40] = 0.8
    cv2.circle(image, (31, 31), 7, 0.65, -1)
    return image


def test_estimate_blur_kernel_smoke() -> None:
    sharp = _synthetic_image()
    motion = np.zeros((5, 5), dtype=np.float32)
    motion[2, 1:4] = 1.0 / 3.0
    blurred = cv2.filter2D(sharp, -1, motion, borderType=cv2.BORDER_REFLECT)
    cfg = DeblurConfig(
        kernel_size=5,
        xk_iter=1,
        dark_patch_size=5,
        max_dark_steps=2,
        max_grad_steps=3,
        fft_workers=1,
    )
    kernel, latent = estimate_blur_kernel(blurred, cfg)
    assert kernel.shape == (5, 5)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-5)
    assert np.isfinite(kernel).all()
    assert latent.shape == blurred.shape


def test_full_rgb_pipeline_smoke() -> None:
    gray = _synthetic_image(40)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    cfg = DeblurConfig(
        kernel_size=5,
        xk_iter=1,
        dark_patch_size=5,
        max_dark_steps=1,
        max_grad_steps=2,
        fft_workers=1,
    )
    result, kernel, interim = deblur_image(rgb, cfg)
    assert result.shape == rgb.shape
    assert kernel.shape == (5, 5)
    assert interim.ndim == 2
    assert np.isfinite(result).all()
