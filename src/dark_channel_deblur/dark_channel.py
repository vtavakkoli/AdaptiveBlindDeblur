from __future__ import annotations

import cv2
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _find_matlab_min_targets(
    padded: np.ndarray,
    minima: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first minimum location using MATLAB's column-major tie order."""
    h, w = minima.shape
    out_y = np.empty(h * w, dtype=np.int32)
    out_x = np.empty(h * w, dtype=np.int32)

    for index in prange(h * w):
        y = index // w
        x = index - y * w
        target = minima[y, x]
        found_y = y
        found_x = x
        found = False
        # MATLAB's min(tmp(:)) visits rows first because linear indexing is
        # column-major. The old Python shortcut visited columns first instead.
        for xx in range(patch_size):
            if found:
                break
            for yy in range(patch_size):
                if padded[y + yy, x + xx] == target:
                    found_y = y + yy
                    found_x = x + xx
                    found = True
                    break
        out_y[index] = found_y
        out_x[index] = found_x
    return out_y, out_x


@njit(cache=True)
def _assign_dark_channel_sequential(
    source: np.ndarray,
    minima: np.ndarray,
    selected: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Reproduce assign_dark_channel_to_pixel.m exactly for grayscale input.

    The order matters. Each window sees modifications made by all earlier windows.
    Replacing this with a bulk write changes the L0 dark-channel prior and was the
    main source of over-sparsification in difficult dark/saturated scenes.
    """
    radius = patch_size // 2
    h, w = source.shape
    padded = np.empty((h + 2 * radius, w + 2 * radius), dtype=np.float32)

    for py in range(h + 2 * radius):
        sy = min(max(py - radius, 0), h - 1)
        for px in range(w + 2 * radius):
            sx = min(max(px - radius, 0), w - 1)
            padded[py, px] = source[sy, sx]

    for y in range(h):
        for x in range(w):
            current_min = padded[y, x]
            for yy in range(patch_size):
                for xx in range(patch_size):
                    value = padded[y + yy, x + xx]
                    if value < current_min:
                        current_min = value

            refined_value = 0.0 if selected[y, x] else minima[y, x]
            if current_min != refined_value:
                index = y * w + x
                padded[target_y[index], target_x[index]] = refined_value

    out = padded[radius : radius + h, radius : radius + w].copy()

    # The MATLAB routine explicitly restores an untouched border after the
    # sequential patch assignment.
    if radius:
        out[:radius, :] = source[:radius, :]
        out[-radius:, :] = source[-radius:, :]
        out[:, :radius] = source[:, :radius]
        out[:, -radius:] = source[:, -radius:]
    return out


def dark_channel(image: np.ndarray, patch_size: int = 35) -> np.ndarray:
    """Compute the MATLAB-equivalent local dark channel."""
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
    """Apply the sequential sparse dark-channel auxiliary-variable projection.

    This is a faithful port of ``dark_channel.m`` plus
    ``assign_dark_channel_to_pixel.m`` for the grayscale latent image used during
    blind PSF estimation. The sequential overlapping-window update is intentional
    and must not be parallelized into independent bulk writes.
    """
    if beta_pixel <= 0:
        raise ValueError("beta_pixel must be > 0")
    arr = np.ascontiguousarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("project_dark_channel expects the grayscale latent image")

    minima = np.ascontiguousarray(dark_channel(arr, patch_size), dtype=np.float32)
    selected = np.ascontiguousarray(minima * minima < (lambda_dark / beta_pixel))

    radius = patch_size // 2
    padded = np.pad(arr, radius, mode="edge")
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    target_y, target_x = _find_matlab_min_targets(padded, minima, patch_size)
    return _assign_dark_channel_sequential(
        arr,
        minima,
        selected,
        target_y,
        target_x,
        patch_size,
    )
