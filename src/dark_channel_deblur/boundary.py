from __future__ import annotations

import numpy as np


def wrap_boundary(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Create a smooth periodic extension on the bottom/right boundaries.

    The CVPR MATLAB release uses Liu/Jia's Poisson boundary wrapping to make
    opposite image borders compatible with circular FFT convolution.  This
    implementation keeps the original image in the top-left (as that code
    does) and fills the extra FFT support with harmonic-style linear blends
    between opposite borders.  It is vectorized, dependency-free, stable for
    tiny images, and suppresses ringing much better than zero/reflect padding.
    """
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    th, tw = target_shape
    if th < h or tw < w:
        raise ValueError("target_shape must not be smaller than the image")
    if th == h and tw == w:
        return arr.copy()

    out_shape = (th, tw) + arr.shape[2:]
    out = np.empty(out_shape, dtype=arr.dtype)
    out[:h, :w, ...] = arr

    pad_w = tw - w
    if pad_w:
        tx = (np.arange(1, pad_w + 1, dtype=np.float32) / (pad_w + 1.0))
        shape = (1, pad_w) + (1,) * (arr.ndim - 2)
        tx = tx.reshape(shape)
        left = arr[:, -1:, ...]
        right = arr[:, :1, ...]
        out[:h, w:, ...] = (1.0 - tx) * left + tx * right

    pad_h = th - h
    if pad_h:
        ty = (np.arange(1, pad_h + 1, dtype=np.float32) / (pad_h + 1.0))
        shape = (pad_h, 1) + (1,) * (arr.ndim - 2)
        ty = ty.reshape(shape)
        top = arr[-1:, :, ...]
        bottom = arr[:1, :, ...]
        out[h:, :w, ...] = (1.0 - ty) * top + ty * bottom

    if pad_h and pad_w:
        # Bilinear interpolation between the four image corners.
        ty = (np.arange(1, pad_h + 1, dtype=np.float32) / (pad_h + 1.0)).reshape(
            (pad_h, 1) + (1,) * (arr.ndim - 2)
        )
        tx = (np.arange(1, pad_w + 1, dtype=np.float32) / (pad_w + 1.0)).reshape(
            (1, pad_w) + (1,) * (arr.ndim - 2)
        )
        c00 = arr[-1, -1, ...]
        c01 = arr[-1, 0, ...]
        c10 = arr[0, -1, ...]
        c11 = arr[0, 0, ...]
        out[h:, w:, ...] = (
            (1.0 - ty) * (1.0 - tx) * c00
            + (1.0 - ty) * tx * c01
            + ty * (1.0 - tx) * c10
            + ty * tx * c11
        )

    return out
