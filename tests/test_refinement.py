from __future__ import annotations

import numpy as np

from dark_channel_deblur import annealed_pnp_refine, extreme_channel_refine, reblur_image


def _sample() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:48, 0:48]
    base = 0.15 + 0.55 * (x / 47.0) + 0.15 * (y / 47.0)
    image = np.stack(
        [base, np.clip(base * 0.85 + 0.05, 0, 1), np.clip(base * 0.70 + 0.10, 0, 1)],
        axis=2,
    ).astype(np.float32)
    image[10:27, 12:34] *= 0.25
    image[30:38, 7:40] = np.clip(image[30:38, 7:40] + 0.22, 0, 1)
    kernel = np.zeros((5, 5), dtype=np.float32)
    kernel[2, 1:4] = np.array([0.25, 0.50, 0.25], dtype=np.float32)
    return image, kernel


def test_reblur_identity_kernel_is_identity() -> None:
    image, _ = _sample()
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1, 1] = 1.0
    reconstructed = reblur_image(image, kernel, workers=1)
    assert reconstructed.shape == image.shape
    assert np.max(np.abs(reconstructed - image)) < 2e-5


def test_annealed_pnp_is_deterministic_finite_and_bounded() -> None:
    image, kernel = _sample()
    blurred = reblur_image(image, kernel, workers=1)
    first = annealed_pnp_refine(
        blurred,
        blurred,
        kernel,
        steps=2,
        candidates=1,
        seed=7,
        workers=1,
    )
    second = annealed_pnp_refine(
        blurred,
        blurred,
        kernel,
        steps=2,
        candidates=1,
        seed=7,
        workers=1,
    )
    assert first.shape == image.shape
    assert np.isfinite(first).all()
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    assert np.allclose(first, second, atol=1e-7)
    assert float(np.sqrt(np.mean((reblur_image(first, kernel, workers=1) - blurred) ** 2))) < 0.08


def test_extreme_channel_refinement_is_finite_and_data_consistent() -> None:
    image, kernel = _sample()
    blurred = reblur_image(image, kernel, workers=1)
    refined = extreme_channel_refine(
        blurred,
        blurred,
        kernel,
        steps=2,
        patch_size=9,
        workers=1,
    )
    assert refined.shape == image.shape
    assert np.isfinite(refined).all()
    assert 0.0 <= float(refined.min()) <= float(refined.max()) <= 1.0
    residual = float(np.sqrt(np.mean((reblur_image(refined, kernel, workers=1) - blurred) ** 2)))
    assert residual < 0.08
