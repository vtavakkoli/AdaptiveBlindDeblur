from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import fft


def psf2otf(psf: np.ndarray, shape: tuple[int, int], workers: int = -1) -> np.ndarray:
    """Convert a spatial PSF to an OTF using MATLAB-compatible centering."""
    psf = np.asarray(psf)
    if psf.ndim != 2:
        raise ValueError("psf must be a 2-D array")
    if psf.shape[0] > shape[0] or psf.shape[1] > shape[1]:
        raise ValueError("PSF cannot be larger than requested OTF shape")

    out = np.zeros(shape, dtype=np.result_type(psf.dtype, np.float32))
    out[: psf.shape[0], : psf.shape[1]] = psf
    out = np.roll(out, -(psf.shape[0] // 2), axis=0)
    out = np.roll(out, -(psf.shape[1] // 2), axis=1)
    return fft.fft2(out, workers=workers)


def otf2psf(otf: np.ndarray, psf_shape: tuple[int, int], workers: int = -1) -> np.ndarray:
    """Inverse of :func:`psf2otf` for a requested finite PSF support."""
    spatial = fft.ifft2(otf, workers=workers).real
    spatial = np.roll(spatial, psf_shape[0] // 2, axis=0)
    spatial = np.roll(spatial, psf_shape[1] // 2, axis=1)
    return spatial[: psf_shape[0], : psf_shape[1]].copy()


@lru_cache(maxsize=1)
def _cho_fft_lut() -> np.ndarray:
    """Generate ``opt_fft_size.m``'s exact 1..4096 lookup table."""
    limit = 4096
    lut = np.zeros(limit + 1, dtype=np.int32)
    e2 = 1
    while e2 <= limit:
        e3 = e2
        while e3 <= limit:
            e5 = e3
            while e5 <= limit:
                e7 = e5
                while e7 <= limit:
                    lut[e7] = e7
                    if e7 * 11 <= limit:
                        lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= limit:
                        lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2

    next_valid = 0
    for index in range(limit, 0, -1):
        if lut[index] != 0:
            next_valid = index
        else:
            lut[index] = next_valid
    return lut


def _cho_fft_size(value: int) -> int:
    if value < 1:
        raise ValueError("FFT dimension must be positive")
    if value <= 4096:
        result = int(_cho_fft_lut()[value])
        if result > 0:
            return result
    # The original helper returns -1 beyond its LUT. Keep the package usable for
    # larger modern inputs while preserving exact release behavior for the full
    # benchmark, whose dimensions are below 4096.
    return int(fft.next_fast_len(value))


def fast_shape(image_shape: tuple[int, int], kernel_shape: tuple[int, int]) -> tuple[int, int]:
    """Return the exact Cho ``opt_fft_size`` support for release-sized images."""
    return (
        _cho_fft_size(image_shape[0] + kernel_shape[0] - 1),
        _cho_fft_size(image_shape[1] + kernel_shape[1] - 1),
    )
