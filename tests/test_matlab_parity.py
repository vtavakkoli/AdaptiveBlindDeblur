from __future__ import annotations

import cv2
import numpy as np

from dark_channel_deblur.boundary import wrap_boundary
from dark_channel_deblur.fft_utils import fast_shape
from dark_channel_deblur.saturation import whyte_deconvolution


def test_cho_fft_sizes_match_release_lookup_examples() -> None:
    # These are deliberately cases where scipy.fft.next_fast_len differs from the
    # original cho_code/opt_fft_size.m lookup table.
    assert fast_shape((1, 1), (13, 13)) == (13, 13)
    assert fast_shape((13, 13), (14, 14)) == (26, 26)
    assert fast_shape((52, 52), (14, 14)) == (65, 65)


def test_liu_boundary_wrap_preserves_source_and_is_harmonic_in_extension() -> None:
    image = np.arange(35, dtype=np.float32).reshape(5, 7) / 34.0
    wrapped = wrap_boundary(image, (10, 12))
    assert wrapped.shape == (10, 12)
    np.testing.assert_array_equal(wrapped[:5, :7], image)
    assert np.isfinite(wrapped).all()

    # The lower-right C region is produced by the minimum-Laplacian Poisson solve.
    c = wrapped[5:, 7:]
    if min(c.shape) >= 3:
        lap = (
            -4.0 * c[1:-1, 1:-1]
            + c[1:-1, 2:]
            + c[1:-1, :-2]
            + c[:-2, 1:-1]
            + c[2:, 1:-1]
        )
        assert float(np.max(np.abs(lap))) < 1e-4


def test_liu_boundary_wrap_handles_one_axis_expansion() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5) / 19.0
    vertical = wrap_boundary(image, (8, 5))
    horizontal = wrap_boundary(image, (4, 9))
    assert vertical.shape == (8, 5)
    assert horizontal.shape == (4, 9)
    np.testing.assert_array_equal(vertical[:4, :5], image)
    np.testing.assert_array_equal(horizontal[:4, :5], image)


def test_whyte_saturated_deconvolution_is_finite_and_native_sized() -> None:
    image = np.zeros((40, 48, 3), dtype=np.float32)
    image[8:32, 10:38] = 0.55
    image[15:26, 20:30] = 1.0
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    kernel = np.zeros((5, 5), dtype=np.float32)
    kernel[2, 1:4] = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    result = whyte_deconvolution(image, kernel, iterations=3, workers=1)
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
