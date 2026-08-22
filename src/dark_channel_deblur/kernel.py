from __future__ import annotations

import cv2
import numpy as np
from scipy import fft

from .fft_utils import otf2psf, psf2otf


def valid_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return same-sized valid horizontal/vertical gradient fields."""
    arr = np.asarray(image, dtype=np.float32)
    return arr[:-1, 1:] - arr[:-1, :-1], arr[1:, :-1] - arr[:-1, :-1]


def threshold_gradients(
    latent: np.ndarray,
    psf_size: int,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select salient gradients using the orientation-balanced CVPR heuristic."""
    px, py = valid_gradients(latent)
    pm = px * px + py * py

    first = threshold is None
    if first:
        # Gradient orientation modulo pi, matching atan(py/px) in the MATLAB release.
        angle = np.arctan2(py, px)
        angle = np.where(angle > np.pi / 2, angle - np.pi, angle)
        angle = np.where(angle < -np.pi / 2, angle + np.pi, angle)
        bins = (
            (angle >= 0) & (angle < np.pi / 4),
            (angle >= np.pi / 4) & (angle <= np.pi / 2),
            (angle >= -np.pi / 4) & (angle < 0),
            (angle >= -np.pi / 2) & (angle < -np.pi / 4),
        )
        count = max(psf_size * 20, 10)
        candidates: list[float] = []
        for mask in bins:
            vals = pm[mask]
            if vals.size:
                k = min(count, vals.size)
                candidates.append(float(np.partition(vals, vals.size - k)[vals.size - k]))
        threshold = min(candidates) if candidates else float(np.percentile(pm, 90))
        if not np.isfinite(threshold) or threshold <= 0:
            positive = pm[pm > 0]
            threshold = float(np.median(positive)) if positive.size else 1e-8

    assert threshold is not None
    mask = pm < threshold
    if np.all(mask) and np.any(pm > 0):
        threshold = min(threshold, float(np.max(pm)) * 0.99)
        mask = pm < threshold
    px = px.copy()
    py = py.copy()
    px[mask] = 0.0
    py[mask] = 0.0
    if not first:
        threshold /= 1.1
    return px, py, float(threshold)


def _apply_kernel_normal_operator(
    x: np.ndarray,
    spectrum: np.ndarray,
    image_shape: tuple[int, int],
    weight: float,
    workers: int,
) -> np.ndarray:
    xf = psf2otf(x, image_shape, workers)
    return otf2psf(spectrum * xf, x.shape, workers) + weight * x


def estimate_psf(
    blurred_x: np.ndarray,
    blurred_y: np.ndarray,
    latent_x: np.ndarray,
    latent_y: np.ndarray,
    weight: float,
    psf_shape: tuple[int, int],
    *,
    workers: int = -1,
    max_iter: int = 20,
    tol: float = 1e-5,
) -> np.ndarray:
    """Estimate a blur PSF with an implicit FFT normal equation + CG."""
    lxf = fft.fft2(latent_x, workers=workers)
    lyf = fft.fft2(latent_y, workers=workers)
    bxf = fft.fft2(blurred_x, workers=workers)
    byf = fft.fft2(blurred_y, workers=workers)
    b = otf2psf(np.conj(lxf) * bxf + np.conj(lyf) * byf, psf_shape, workers)
    spectrum = np.conj(lxf) * lxf + np.conj(lyf) * lyf

    x = np.full(psf_shape, 1.0 / np.prod(psf_shape), dtype=np.float64)
    r = b - _apply_kernel_normal_operator(x, spectrum, blurred_x.shape, weight, workers)
    p = r.copy()
    rsold = float(np.vdot(r, r).real)
    for _ in range(max_iter):
        ap = _apply_kernel_normal_operator(p, spectrum, blurred_x.shape, weight, workers)
        denom = float(np.vdot(p, ap).real)
        if abs(denom) < 1e-20:
            break
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * ap
        rsnew = float(np.vdot(r, r).real)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / max(rsold, 1e-30)) * p
        rsold = rsnew

    peak = float(np.max(x))
    if peak > 0:
        x[x < peak * 0.05] = 0.0
    x[x < 0] = 0.0
    total = float(x.sum())
    if total <= 1e-12:
        x[:] = 0.0
        x[psf_shape[0] // 2, psf_shape[1] // 2] = 1.0
    else:
        x /= total
    return x.astype(np.float32)


def prune_kernel(kernel: np.ndarray, min_component_mass: float = 0.1) -> np.ndarray:
    k = np.maximum(np.asarray(kernel, dtype=np.float32), 0.0).copy()
    mask = (k > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    for label in range(1, n):
        component = labels == label
        if float(k[component].sum()) < min_component_mass:
            k[component] = 0.0
    total = float(k.sum())
    if total > 0:
        k /= total
    return k


def adjust_psf_center(kernel: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(kernel, dtype=np.float32), 0.0)
    total = float(k.sum())
    if total <= 0:
        return k.copy()
    yy, xx = np.indices(k.shape, dtype=np.float32)
    cy = float((k * yy).sum() / total)
    cx = float((k * xx).sum() / total)
    ty = (k.shape[0] - 1) / 2.0
    tx = (k.shape[1] - 1) / 2.0
    sy = int(round(ty - cy))
    sx = int(round(tx - cx))
    out = np.zeros_like(k)

    src_y0 = max(0, -sy)
    src_y1 = min(k.shape[0], k.shape[0] - sy)
    src_x0 = max(0, -sx)
    src_x1 = min(k.shape[1], k.shape[1] - sx)
    dst_y0 = src_y0 + sy
    dst_y1 = src_y1 + sy
    dst_x0 = src_x0 + sx
    dst_x1 = src_x1 + sx
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = k[src_y0:src_y1, src_x0:src_x1]
    total = float(out.sum())
    return out / total if total > 0 else k / float(k.sum())


def init_kernel(size: int) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    row = size // 2
    left = max(0, row - 1)
    k[row, left : left + 2] = 0.5
    return k


def resize_kernel(kernel: np.ndarray, scale: float, target_size: int) -> np.ndarray:
    new_h = max(1, int(round(kernel.shape[0] * scale)))
    new_w = max(1, int(round(kernel.shape[1] * scale)))
    k = cv2.resize(kernel, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    k = np.maximum(k, 0.0)

    # Center crop/pad to requested support; subsequent centroid correction handles drift.
    out = np.zeros((target_size, target_size), dtype=np.float32)
    h, w = k.shape
    src_y0 = max(0, (h - target_size) // 2)
    src_x0 = max(0, (w - target_size) // 2)
    dst_y0 = max(0, (target_size - h) // 2)
    dst_x0 = max(0, (target_size - w) // 2)
    hh = min(h, target_size)
    ww = min(w, target_size)
    out[dst_y0 : dst_y0 + hh, dst_x0 : dst_x0 + ww] = k[src_y0 : src_y0 + hh, src_x0 : src_x0 + ww]
    total = float(out.sum())
    if total > 0:
        out /= total
    return out
