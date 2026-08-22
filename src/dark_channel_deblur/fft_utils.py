from __future__ import annotations

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


def fast_shape(image_shape: tuple[int, int], kernel_shape: tuple[int, int]) -> tuple[int, int]:
    """Return FFT-friendly dimensions large enough for image + kernel - 1."""
    return (
        fft.next_fast_len(image_shape[0] + kernel_shape[0] - 1),
        fft.next_fast_len(image_shape[1] + kernel_shape[1] - 1),
    )
