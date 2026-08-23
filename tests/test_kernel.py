from __future__ import annotations

import numpy as np

from dark_channel_deblur.kernel import (
    adjust_psf_center,
    init_kernel,
    prune_kernel,
    resize_kernel,
    threshold_gradients,
)


def test_adjust_psf_center_moves_mass_toward_center() -> None:
    k = np.zeros((7, 7), dtype=np.float32)
    k[1, 1] = 1.0
    centered = adjust_psf_center(k)
    assert np.unravel_index(np.argmax(centered), centered.shape) == (3, 3)
    np.testing.assert_allclose(centered.sum(), 1.0, atol=1e-6)


def test_prune_kernel_removes_tiny_component() -> None:
    k = np.zeros((7, 7), dtype=np.float32)
    k[3, 3] = 0.95
    k[0, 0] = 0.05
    out = prune_kernel(k, min_component_mass=0.1)
    assert out[0, 0] == 0
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


def test_init_kernel_matches_matlab_release_coordinates() -> None:
    actual = init_kernel(25)
    expected = np.zeros((25, 25), dtype=np.float32)
    # blind_deconv.m uses 1-based row/columns 12 and 12:13 for a 25px PSF.
    expected[11, 11:13] = 0.5
    np.testing.assert_array_equal(actual, expected)


def test_resize_kernel_preserves_mass_and_requested_support() -> None:
    kernel = np.zeros((7, 7), dtype=np.float32)
    kernel[1:6, 2:5] = np.array(
        [
            [0.01, 0.02, 0.01],
            [0.03, 0.08, 0.04],
            [0.05, 0.20, 0.08],
            [0.04, 0.15, 0.06],
            [0.02, 0.10, 0.03],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    resized = resize_kernel(kernel, np.sqrt(2.0), 9)
    assert resized.shape == (9, 9)
    assert np.all(resized >= 0)
    np.testing.assert_allclose(resized.sum(), 1.0, atol=1e-6)


def test_threshold_gradients_produces_nonzero_salient_edges_on_textured_image() -> None:
    rng = np.random.default_rng(29)
    image = rng.random((128, 128), dtype=np.float32)
    px, py, threshold = threshold_gradients(image, 5)
    assert threshold > 0
    assert np.count_nonzero(px) > 0
    assert np.count_nonzero(py) > 0
    assert px.shape == py.shape == (127, 127)


def test_single_orientation_can_keep_zero_matlab_histogram_threshold() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[:, 16:] = 1.0
    px, py, threshold = threshold_gradients(image, 5)
    # threshold_pxpy_v1 takes the minimum tail count across four orientation bins;
    # a purely vertical synthetic edge does not populate all four bins.
    assert threshold == 0.0
    assert np.count_nonzero(px) > 0
    assert np.count_nonzero(py) == 0


def test_estimate_psf_recovers_synthetic_kernel() -> None:
    from scipy import fft

    from dark_channel_deblur.fft_utils import psf2otf
    from dark_channel_deblur.kernel import estimate_psf, valid_gradients

    rng = np.random.default_rng(101)
    latent = rng.random((48, 48), dtype=np.float32)
    lx, ly = valid_gradients(latent)
    true_kernel = np.zeros((5, 5), dtype=np.float32)
    true_kernel[2, 1:4] = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    otf = psf2otf(true_kernel, lx.shape, workers=1)
    bx = fft.ifft2(fft.fft2(lx) * otf).real.astype(np.float32)
    by = fft.ifft2(fft.fft2(ly) * otf).real.astype(np.float32)

    estimated = estimate_psf(
        bx,
        by,
        lx,
        ly,
        weight=1e-4,
        psf_shape=(5, 5),
        workers=1,
        max_iter=60,
        tol=1e-8,
    )
    np.testing.assert_allclose(estimated, true_kernel, atol=2e-3)
