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


def _matlab_projection_reference(
    image: np.ndarray,
    lambda_dark: float,
    beta_pixel: float,
    patch: int,
) -> np.ndarray:
    """Literal small-image translation of the two MATLAB dark-channel routines."""
    source = np.asarray(image, dtype=np.float32)
    h, w = source.shape
    radius = patch // 2
    padded_source = np.pad(source, radius, mode="edge")

    minima = np.empty_like(source)
    indices = np.empty((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            window = padded_source[y : y + patch, x : x + patch]
            minima[y, x] = window.min()
            # tmp(:) is MATLAB/Fortran column-major linear indexing.
            indices[y, x] = int(np.argmin(window.reshape(-1, order="F")))

    refined = minima.copy()
    refined[refined * refined < lambda_dark / beta_pixel] = 0.0

    work = padded_source.copy()
    for y in range(h):
        for x in range(w):
            window = work[y : y + patch, x : x + patch].copy()
            if float(window.min()) != float(refined[y, x]):
                linear = int(indices[y, x])
                yy = linear % patch
                xx = linear // patch
                window[yy, xx] = refined[y, x]
            work[y : y + patch, x : x + patch] = window

    out = work[radius : radius + h, radius : radius + w].copy()
    if radius:
        out[:radius, :] = source[:radius, :]
        out[-radius:, :] = source[-radius:, :]
        out[:, :radius] = source[:, :radius]
        out[:, -radius:] = source[:, -radius:]
    return out


def test_dark_channel_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((11, 13), dtype=np.float32)
    actual = dark_channel(image, 5)
    expected = _brute_dark(image, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_project_dark_channel_matches_matlab_sequential_reference() -> None:
    rng = np.random.default_rng(17)
    image = rng.random((13, 15), dtype=np.float32)
    actual = project_dark_channel(
        image,
        lambda_dark=4e-3,
        beta_pixel=0.10,
        patch_size=5,
    )
    expected = _matlab_projection_reference(image, 4e-3, 0.10, 5)
    np.testing.assert_array_equal(actual, expected)


def test_project_dark_channel_preserves_shape_boundary_and_finiteness() -> None:
    rng = np.random.default_rng(3)
    image = rng.random((16, 18), dtype=np.float32)
    out = project_dark_channel(image, lambda_dark=4e-3, beta_pixel=0.1, patch_size=5)
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:2], image[:2])
    np.testing.assert_array_equal(out[-2:], image[-2:])
    np.testing.assert_array_equal(out[:, :2], image[:, :2])
    np.testing.assert_array_equal(out[:, -2:], image[:, -2:])
