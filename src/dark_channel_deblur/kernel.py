from __future__ import annotations

import cv2
import numpy as np
from scipy import fft

from .fft_utils import otf2psf, psf2otf


def valid_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gradients equivalent to MATLAB conv2(..., dx/dy, 'valid')."""
    arr = np.asarray(image, dtype=np.float32)
    # MATLAB conv2 performs convolution (not correlation), so these have the
    # opposite sign to a forward difference. Both blurred and latent gradients
    # use the same convention, which is important for parity/debugging.
    return arr[:-1, :-1] - arr[:-1, 1:], arr[:-1, :-1] - arr[1:, :-1]


def _histc_tail(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Reproduce cumsum(flipud(histc(values, steps))) for non-negative values."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    counts = np.zeros(steps.size, dtype=np.int64)
    if vals.size == 0:
        return counts
    indices = np.searchsorted(steps, vals, side="right") - 1
    valid = (indices >= 0) & (indices < steps.size) & (vals <= steps[-1])
    if np.any(valid):
        counts += np.bincount(indices[valid], minlength=steps.size)[: steps.size]
    return np.cumsum(counts[::-1])


def threshold_gradients(
    latent: np.ndarray,
    psf_size: int,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Faithful port of ``threshold_pxpy_v1.m`` from the MATLAB release."""
    px, py = valid_gradients(latent)
    pm = px * px + py * py

    first = threshold is None
    if first:
        with np.errstate(divide="ignore", invalid="ignore"):
            pd = np.arctan(py / px)
        steps = np.arange(0.0, 2.0 + 0.00003, 0.00006, dtype=np.float64)
        # MATLAB's colon 0:0.00006:2 stops at the last representable step <= 2.
        steps = steps[steps <= 2.0 + 1e-12]
        bins = (
            (pd >= 0) & (pd < np.pi / 4),
            (pd >= np.pi / 4) & (pd < np.pi / 2),
            (pd >= -np.pi / 4) & (pd < 0),
            (pd >= -np.pi / 2) & (pd < -np.pi / 4),
        )
        tails = np.stack([_histc_tail(pm[mask], steps) for mask in bins], axis=0)
        required = max(int(psf_size) * 20, 10)
        threshold = 0.0
        eligible = np.flatnonzero(np.min(tails, axis=0) >= required)
        if eligible.size:
            t = int(eligible[0])
            threshold = float(steps[-1 - t])

    assert threshold is not None
    mask = pm < threshold
    max_pm = float(np.max(pm)) if pm.size else 0.0
    while np.all(mask) and max_pm > 0.0:
        threshold *= 0.81
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
    """Estimate a blur PSF with the release's FFT normal equation + CG solve."""
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
    total = float(x.sum())
    if abs(total) <= 1e-12:
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
    """Match the intentionally off-centre two-tap initialization in blind_deconv.m."""
    k = np.zeros((size, size), dtype=np.float32)
    row = size // 2 - 1
    left = size // 2 - 1
    k[row, left : left + 2] = 0.5
    return k


def _fix_kernel_size(kernel: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Port Levin's mass-aware ``fixsize`` helper used by ``resizeKer``."""
    k = np.asarray(kernel, dtype=np.float32)
    while k.shape != (rows, cols):
        h, w = k.shape
        if h > rows:
            sums = np.sum(k, axis=1)
            k = k[1:, :] if sums[0] < sums[-1] else k[:-1, :]
        elif h < rows:
            sums = np.sum(k, axis=1)
            padded = np.zeros((h + 1, w), dtype=k.dtype)
            if sums[0] < sums[-1]:
                padded[:h, :] = k
            else:
                padded[1:, :] = k
            k = padded

        h, w = k.shape
        if w > cols:
            sums = np.sum(k, axis=0)
            k = k[:, 1:] if sums[0] < sums[-1] else k[:, :-1]
        elif w < cols:
            sums = np.sum(k, axis=0)
            padded = np.zeros((h, w + 1), dtype=k.dtype)
            if sums[0] < sums[-1]:
                padded[:, :w] = k
            else:
                padded[:, 1:] = k
            k = padded
    return k


def resize_kernel(kernel: np.ndarray, scale: float, target_size: int) -> np.ndarray:
    """Resize a PSF then apply the MATLAB release's mass-aware support correction."""
    new_h = max(1, int(np.ceil(kernel.shape[0] * scale)))
    new_w = max(1, int(np.ceil(kernel.shape[1] * scale)))
    k = cv2.resize(
        np.asarray(kernel, dtype=np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    k = np.maximum(k, 0.0)
    k = _fix_kernel_size(k, target_size, target_size)
    total = float(k.sum())
    if total > 0:
        k /= total
    return k.astype(np.float32)
