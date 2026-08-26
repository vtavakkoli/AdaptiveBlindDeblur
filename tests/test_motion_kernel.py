from __future__ import annotations

import cv2
import numpy as np
from scipy import fft

from dark_channel_deblur.config import DeblurConfig
from dark_channel_deblur.fft_utils import psf2otf
from dark_channel_deblur.kernel import estimate_psf, valid_gradients
from dark_channel_deblur.motion_kernel import _infer_motion_corridor, estimate_motion_constrained_psf


def _curved_motion_kernel(size: int = 17) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    for t in np.linspace(0.0, 1.0, 48):
        point = (
            (1.0 - t) ** 2 * np.array([-5.5, -3.5])
            + 2.0 * (1.0 - t) * t * np.array([0.0, 4.5])
            + t**2 * np.array([5.5, 2.5])
        )
        x = center + float(point[0])
        y = center + float(point[1])
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        dx = x - x0
        dy = y - y0
        for oy, wy in ((0, 1.0 - dy), (1, dy)):
            for ox, wx in ((0, 1.0 - dx), (1, dx)):
                yy = y0 + oy
                xx = x0 + ox
                if 0 <= yy < size and 0 <= xx < size:
                    kernel[yy, xx] += np.float32(wy * wx)
    kernel = cv2.GaussianBlur(kernel, (0, 0), sigmaX=0.55, sigmaY=0.55)
    kernel /= kernel.sum()
    return kernel


def test_motion_corridor_is_connected_compact_and_keeps_curve() -> None:
    rng = np.random.default_rng(17)
    clean = _curved_motion_kernel()
    noisy = clean + rng.uniform(0.0, 0.003, clean.shape).astype(np.float32)
    for y, x in ((0, 1), (1, 15), (15, 2), (16, 14)):
        noisy[y, x] += 0.02
    noisy /= noisy.sum()

    corridor = _infer_motion_corridor(noisy, corridor_radius=2)
    count, _ = cv2.connectedComponents(corridor.astype(np.uint8), connectivity=8)

    assert count == 2  # background + one connected admissible trajectory
    assert float(corridor.mean()) < 0.40
    assert float(clean[corridor].sum()) > 0.90


def test_motion_constrained_estimator_rejects_off_trajectory_gradient_noise() -> None:
    rng = np.random.default_rng(2026)
    latent = rng.random((64, 64), dtype=np.float32)
    latent_x, latent_y = valid_gradients(latent)
    true_kernel = _curved_motion_kernel()
    otf = psf2otf(true_kernel, latent_x.shape, workers=1)
    blurred_x = fft.ifft2(fft.fft2(latent_x) * otf).real.astype(np.float32)
    blurred_y = fft.ifft2(fft.fft2(latent_y) * otf).real.astype(np.float32)
    blurred_x += rng.normal(0.0, 0.02, blurred_x.shape).astype(np.float32)
    blurred_y += rng.normal(0.0, 0.02, blurred_y.shape).astype(np.float32)

    free = estimate_psf(
        blurred_x,
        blurred_y,
        latent_x,
        latent_y,
        weight=1e-3,
        psf_shape=true_kernel.shape,
        workers=1,
        max_iter=25,
        tol=1e-8,
        peak_fraction=0.0,
    )
    free = np.maximum(free, 0.0)
    free /= free.sum()
    constrained = estimate_motion_constrained_psf(
        blurred_x,
        blurred_y,
        latent_x,
        latent_y,
        weight=1e-3,
        psf_shape=true_kernel.shape,
        workers=1,
        warm_start_iter=25,
        projected_iter=36,
        tol=1e-8,
        corridor_radius=2,
    )

    np.testing.assert_allclose(constrained.sum(), 1.0, atol=1e-6)
    assert float(constrained.min()) >= 0.0
    free_l1 = float(np.abs(free - true_kernel).sum())
    constrained_l1 = float(np.abs(constrained - true_kernel).sum())
    assert constrained_l1 < free_l1 * 0.75


def test_motion_config_validation() -> None:
    DeblurConfig(kernel_model="motion-trajectory", motion_corridor_radius=2, motion_pgd_steps=8).validate()

    for kwargs in (
        {"kernel_model": "unknown"},
        {"motion_corridor_radius": 0},
        {"motion_pgd_steps": 0},
    ):
        try:
            DeblurConfig(**kwargs).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid motion configuration: {kwargs}")
