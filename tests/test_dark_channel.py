from __future__ import annotations

import numpy as np

from dark_channel_deblur.dark_channel import dark_channel, project_dark_channel


def _brute_dark(image: np.ndarray, patch: int) -> np.ndarray:
    r = patch // 2
    padded = np.pad(image, r, mode="edge")
    out = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = padded[y : y + patch, x : x + patch].min()
    return out


def test_dark_channel_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((11, 13), dtype=np.float32)
    actual = dark_channel(image, 5)
    expected = _brute_dark(image, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_project_dark_channel_preserves_shape_and_finiteness() -> None:
    rng = np.random.default_rng(3)
    image = rng.random((16, 18), dtype=np.float32)
    out = project_dark_channel(image, lambda_dark=4e-3, beta_pixel=0.1, patch_size=5)
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert np.count_nonzero(out == 0) >= np.count_nonzero(image == 0)
