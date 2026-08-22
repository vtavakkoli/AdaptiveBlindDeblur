from __future__ import annotations

import cv2
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _find_first_min_targets(
    padded: np.ndarray,
    minima: np.ndarray,
    selected: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate one row-major minimum per selected window in parallel."""
    h, w = minima.shape
    out_y = np.full(h * w, -1, dtype=np.int32)
    out_x = np.full(h * w, -1, dtype=np.int32)
    channels = 1 if padded.ndim == 2 else padded.shape[2]

    for idx in prange(h * w):
        y = idx // w
        x = idx - y * w
        if not selected[y, x]:
            continue
        target = minima[y, x]
        found = False
        for yy in range(patch_size):
            if found:
                break
            for xx in range(patch_size):
                if channels == 1:
                    if padded[y + yy, x + xx] <= target + 1e-7:
                        out_y[idx] = y + yy
                        out_x[idx] = x + xx
                        found = True
                        break
                else:
                    for cc in range(channels):
                        if padded[y + yy, x + xx, cc] <= target + 1e-7:
                            out_y[idx] = y + yy
                            out_x[idx] = x + xx
                            found = True
                            break
                    if found:
                        break
    return out_y, out_x


def dark_channel(image: np.ndarray, patch_size: int = 35) -> np.ndarray:
    """Compute the local dark channel using OpenCV's optimized erosion."""
    if patch_size % 2 == 0 or patch_size < 1:
        raise ValueError("patch_size must be a positive odd integer")
    arr = np.asarray(image, dtype=np.float32)
    base = arr if arr.ndim == 2 else np.min(arr, axis=2)
    kernel = np.ones((patch_size, patch_size), dtype=np.uint8)
    return cv2.erode(base, kernel, borderType=cv2.BORDER_REPLICATE)


def project_dark_channel(
    image: np.ndarray,
    lambda_dark: float,
    beta_pixel: float,
    patch_size: int = 35,
) -> np.ndarray:
    """Fast auxiliary-variable projection for the L0 dark-channel prior.

    Windows whose dark-channel value is below sqrt(lambda/beta) select one
    local minimum. Those selected pixels are zeroed in one bulk projection.
    This avoids the MATLAB implementation's repeated patch copying and makes
    the dominant prior update practical in Python.
    """
    arr = np.ascontiguousarray(image, dtype=np.float32)
    minima = dark_channel(arr, patch_size)
    selected = np.ascontiguousarray(minima * minima < (lambda_dark / beta_pixel))
    if not np.any(selected):
        return arr.copy()

    radius = patch_size // 2
    if arr.ndim == 2:
        padded = np.pad(arr, radius, mode="edge")
    else:
        padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    ys, xs = _find_first_min_targets(padded, minima, selected, patch_size)

    out = arr.copy()
    valid = ys >= 0
    yy = ys[valid] - radius
    xx = xs[valid] - radius
    inside = (yy >= radius) & (yy < arr.shape[0] - radius) & (xx >= radius) & (xx < arr.shape[1] - radius)
    yy = yy[inside]
    xx = xx[inside]
    if arr.ndim == 2:
        out[yy, xx] = 0.0
    else:
        # Select the darkest channel at each mapped pixel.
        cc = np.argmin(out[yy, xx, :], axis=1)
        out[yy, xx, cc] = 0.0
    return out
