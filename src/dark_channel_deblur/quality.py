from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ArtifactDiagnostics:
    edge_ratio: float
    noise_ratio: float
    highpass_ratio: float
    clipping_growth: float


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


def edge_energy(image: np.ndarray) -> float:
    g = _gray(image)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def noise_mad(image: np.ndarray) -> float:
    lap = cv2.Laplacian(_gray(image), cv2.CV_32F)
    median = float(np.median(lap))
    return float(np.median(np.abs(lap - median)))


def highpass_rms(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    smooth = cv2.GaussianBlur(arr, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
    return float(np.sqrt(np.mean((arr - smooth) ** 2)))


def clipping_fraction(image: np.ndarray, margin: float = 1.0 / 255.0) -> float:
    arr = np.asarray(image, dtype=np.float32)
    return float(np.mean((arr <= margin) | (arr >= 1.0 - margin)))


def artifact_diagnostics(observed: np.ndarray, candidate: np.ndarray) -> ArtifactDiagnostics:
    obs = np.asarray(observed, dtype=np.float32)
    out = np.asarray(candidate, dtype=np.float32)
    return ArtifactDiagnostics(
        edge_ratio=edge_energy(out) / max(edge_energy(obs), 0.02),
        noise_ratio=noise_mad(out) / max(noise_mad(obs), 0.003),
        highpass_ratio=highpass_rms(out) / max(highpass_rms(obs), 0.005),
        clipping_growth=max(0.0, clipping_fraction(out) - clipping_fraction(obs)),
    )


def ripple_risk(diag: ArtifactDiagnostics, *, kernel_size: int) -> bool:
    """Detect structured over-deconvolution without using a reference image.

    Long blind kernels are especially prone to a false solution that reblurs well
    but creates repeated contours, zipper texture, and clipped oscillations. The
    detector uses two complementary signatures: clipping plus strong amplification,
    or severe simultaneous edge/high-pass/noise growth even when later regularization
    has already reduced clipping. Sharp-but-clean results are deliberately preserved.
    """
    if diag.clipping_growth >= 0.065:
        return True
    if kernel_size >= 65 and diag.clipping_growth >= 0.025:
        if diag.edge_ratio >= 2.50 or diag.highpass_ratio >= 2.50:
            return True
    if kernel_size >= 65:
        return (
            diag.edge_ratio >= 2.60
            and diag.highpass_ratio >= 3.50
            and diag.noise_ratio >= 2.00
        )
    return False


def saturation_instability(diag: ArtifactDiagnostics) -> bool:
    """Flag an over-iterated saturated-scene reconstruction.

    Saturation-aware Richardson-Lucy can amplify sensor noise and edge halos late
    in the iteration sequence. Requiring both edge and noise growth keeps the
    original full-iteration result for well-behaved saturated scenes.
    """
    return diag.edge_ratio > 2.10 and diag.noise_ratio > 2.20


def saturation_checkpoint_safe(diag: ArtifactDiagnostics) -> bool:
    """Return whether an RL checkpoint remains inside conservative artifact budgets."""
    return (
        diag.edge_ratio <= 1.80
        and diag.noise_ratio <= 1.70
        and diag.highpass_ratio <= 1.70
        and diag.clipping_growth <= 0.020
    )


def artifact_severity(diag: ArtifactDiagnostics) -> float:
    """Reference-free scalar used only to rank already-suspicious candidates."""
    return (
        max(0.0, diag.edge_ratio - 1.80)
        + 1.25 * max(0.0, diag.noise_ratio - 1.70)
        + max(0.0, diag.highpass_ratio - 1.70)
        + 25.0 * diag.clipping_growth
    )


def restoration_score(
    observed: np.ndarray,
    candidate: np.ndarray,
    reblurred: np.ndarray,
) -> tuple[float, ArtifactDiagnostics]:
    """Return a blind quality score balancing blur fidelity and artifact growth.

    The score deliberately avoids legacy/ground-truth pixels. It penalizes a common
    blind-deconvolution failure mode where a latent image reblurs well but contains
    duplicated contours, clipped highlights, ringing, or excessive high frequencies.
    Lower is better.
    """
    obs = np.asarray(observed, dtype=np.float32)
    out = np.asarray(candidate, dtype=np.float32)
    pred = np.asarray(reblurred, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    diag = artifact_diagnostics(obs, out)

    penalty = 0.0
    penalty += 0.006 * max(0.0, diag.noise_ratio - 1.45) ** 2
    penalty += 0.005 * max(0.0, diag.edge_ratio - 3.0) ** 2
    penalty += 0.004 * max(0.0, diag.highpass_ratio - 4.0) ** 2
    penalty += 0.18 * diag.clipping_growth
    return rmse + penalty, diag


def kernel_component_count(kernel: np.ndarray) -> int:
    k = np.maximum(np.asarray(kernel, dtype=np.float32), 0.0)
    peak = float(k.max())
    if peak <= 0:
        return 0
    mask = (k >= peak * 0.05).astype(np.uint8)
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count - 1))


def should_retry_kernel(
    observed: np.ndarray,
    restored: np.ndarray,
    kernel: np.ndarray,
    *,
    kernel_size: int,
    blind_score: float,
) -> bool:
    """Flag suspicious blind solutions for a second, gradient-only PSF estimate."""
    diag = artifact_diagnostics(observed, restored)
    components = kernel_component_count(kernel)

    if components >= 3:
        return True
    if ripple_risk(diag, kernel_size=kernel_size):
        return True
    if diag.clipping_growth > 0.07:
        return True
    if diag.edge_ratio > 3.0 and diag.highpass_ratio > 4.0:
        return True
    # Only retry genuinely under-restored long-support solutions. A previous 1.8x
    # threshold incorrectly retried strong but clean long-motion cases.
    if kernel_size >= 65 and diag.edge_ratio < 1.15 and blind_score > 0.025:
        return True
    if kernel_size >= 65 and blind_score > 0.045:
        return True
    return False
