from __future__ import annotations

import numpy as np
from scipy import fft


def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """Solve the zero-Laplacian interior with fixed boundary samples.

    This is the DST Poisson solve used by ``wrap_boundary_liu.m``. Using an
    orthonormal DST-I pair is algebraically equivalent to the MATLAB dst/idst
    pair because the forward/inverse scaling cancels around the eigenvalue
    division.
    """
    boundary = np.asarray(boundary_image, dtype=np.float64).copy()
    h, w = boundary.shape
    if h <= 2 or w <= 2:
        return boundary

    boundary_only = boundary.copy()
    boundary_only[1:-1, 1:-1] = 0.0
    f_bp = np.zeros_like(boundary_only)
    f_bp[1:-1, 1:-1] = (
        -4.0 * boundary_only[1:-1, 1:-1]
        + boundary_only[1:-1, 2:]
        + boundary_only[1:-1, :-2]
        + boundary_only[:-2, 1:-1]
        + boundary_only[2:, 1:-1]
    )
    rhs = -f_bp[1:-1, 1:-1]
    transformed = fft.dstn(rhs, type=1, norm="ortho")

    x = np.arange(1, w - 1, dtype=np.float64)[None, :]
    y = np.arange(1, h - 1, dtype=np.float64)[:, None]
    denominator = (
        2.0 * np.cos(np.pi * x / (w - 1))
        - 2.0
        + 2.0 * np.cos(np.pi * y / (h - 1))
        - 2.0
    )
    interior = fft.idstn(transformed / denominator, type=1, norm="ortho")

    result = boundary_only
    result[1:-1, 1:-1] = interior
    return result


def _wrap_channel(channel: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Faithful NumPy/SciPy translation of ``wrap_boundary_liu.m``."""
    image = np.asarray(channel, dtype=np.float64)
    h, w = image.shape
    th, tw = target_shape
    h_extra = th - h
    w_extra = tw - w
    if h_extra == 0 and w_extra == 0:
        return image.copy()

    # A: vertical transition under the source image.
    a_region = np.zeros((h_extra + 2, w), dtype=np.float64)
    a_region[0, :] = image[-1, :]
    a_region[-1, :] = image[0, :]
    if h_extra:
        if h_extra == 1:
            blend = np.array([0.0], dtype=np.float64)
        else:
            blend = np.arange(h_extra, dtype=np.float64) / (h_extra - 1.0)
        a_region[1:-1, 0] = (1.0 - blend) * a_region[0, 0] + blend * a_region[-1, 0]
        a_region[1:-1, -1] = (1.0 - blend) * a_region[0, -1] + blend * a_region[-1, -1]
    a_region = _solve_min_laplacian(a_region)

    # B: horizontal transition to the right of the source image.
    b_region = np.zeros((h, w_extra + 2), dtype=np.float64)
    b_region[:, 0] = image[:, -1]
    b_region[:, -1] = image[:, 0]
    if w_extra:
        if w_extra == 1:
            blend = np.array([0.0], dtype=np.float64)
        else:
            blend = np.arange(w_extra, dtype=np.float64) / (w_extra - 1.0)
        b_region[0, 1:-1] = (1.0 - blend) * b_region[0, 0] + blend * b_region[0, -1]
        b_region[-1, 1:-1] = (1.0 - blend) * b_region[-1, 0] + blend * b_region[-1, -1]
    b_region = _solve_min_laplacian(b_region)

    # C: lower-right corner transition constrained by A and B.
    c_region = np.zeros((h_extra + 2, w_extra + 2), dtype=np.float64)
    c_region[0, :] = b_region[-1, :]
    c_region[-1, :] = b_region[0, :]
    c_region[:, 0] = a_region[:, -1]
    c_region[:, -1] = a_region[:, 0]
    c_region = _solve_min_laplacian(c_region)

    a = a_region[:-2, :]
    b = b_region[:, 1:-1]
    c = c_region[1:-1, 1:-1]
    return np.block([[image, b], [a, c]])


def wrap_boundary(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Create the minimum-Laplacian periodic extension used by the MATLAB code."""
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    th, tw = target_shape
    if th < h or tw < w:
        raise ValueError("target_shape must not be smaller than the image")
    if th == h and tw == w:
        return arr.copy()

    if arr.ndim == 2:
        wrapped = _wrap_channel(arr, target_shape)
    elif arr.ndim == 3:
        wrapped = np.stack(
            [_wrap_channel(arr[..., channel], target_shape) for channel in range(arr.shape[2])],
            axis=2,
        )
    else:
        raise ValueError("image must be HxW or HxWxC")
    return wrapped.astype(arr.dtype, copy=False)
