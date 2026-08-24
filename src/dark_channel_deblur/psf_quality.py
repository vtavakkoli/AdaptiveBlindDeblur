from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PSFPlausibility:
    """Reference-free structural diagnostics for a normalized motion PSF."""

    largest_component_mass: float
    secondary_component_mass: float
    anisotropy: float
    off_axis_mass: float
    weak_line_mass: float

    @property
    def score(self) -> float:
        """Lower is more physically plausible; complex connected paths are allowed."""
        anisotropic_off_axis = self.off_axis_mass if self.anisotropy >= 4.0 else 0.25 * self.off_axis_mass
        return (
            0.75 * anisotropic_off_axis
            + 1.25 * self.weak_line_mass
            + 0.20 * max(0.0, self.secondary_component_mass - 0.25)
        )


def _normalize(kernel: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(kernel, dtype=np.float32), 0.0).copy()
    total = float(k.sum())
    if total > 0:
        k /= total
    return k


def _component_masses(kernel: np.ndarray, threshold_fraction: float = 0.015) -> list[float]:
    k = _normalize(kernel)
    peak = float(k.max())
    if peak <= 0:
        return []
    mask = (k >= peak * threshold_fraction).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    masses = [float(k[labels == label].sum()) for label in range(1, count)]
    return sorted(masses, reverse=True)


def _principal_axis(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return centroid, dominant direction and weighted covariance anisotropy.

    The axis is estimated from the higher-confidence PSF core so a weak long spur
    cannot become the dominant orientation by sheer geometric extent.
    """
    k = _normalize(kernel)
    peak = float(k.max())
    if peak <= 0:
        return np.zeros(2, dtype=np.float64), np.array([1.0, 0.0]), 1.0

    core = k >= peak * 0.12
    if int(core.sum()) < 3:
        core = k > 0
    yy, xx = np.nonzero(core)
    weights = k[yy, xx].astype(np.float64)
    if weights.size < 2 or float(weights.sum()) <= 0:
        center = np.array([(k.shape[1] - 1) / 2.0, (k.shape[0] - 1) / 2.0])
        return center, np.array([1.0, 0.0]), 1.0

    coords = np.column_stack([xx, yy]).astype(np.float64)
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    covariance = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-12)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]]
    major /= max(float(np.linalg.norm(major)), 1e-12)
    largest = max(float(values[order[0]]), 1e-9)
    smallest = max(float(values[order[-1]]), 1e-9)
    return center, major, largest / smallest


def _weak_line_mask(kernel: np.ndarray) -> np.ndarray:
    """Detect weak row/column spurs without deleting dense/curved PSF structure."""
    k = _normalize(kernel)
    peak = float(k.max())
    result = np.zeros(k.shape, dtype=bool)
    if peak <= 0:
        return result

    support = k >= peak * 0.015
    strong = k >= peak * 0.10
    h, w = k.shape
    total = max(float(k.sum()), 1e-12)

    # A spurious branch typically spans a large fraction of a row/column, contains
    # almost no high-confidence PSF pixels, and carries little total probability.
    for y in range(h):
        xs = np.flatnonzero(support[y])
        if xs.size < max(5, int(0.28 * w)):
            continue
        span = int(xs[-1] - xs[0] + 1)
        mass = float(k[y, xs].sum()) / total
        strong_count = int(np.count_nonzero(strong[y, xs]))
        if span >= int(0.38 * w) and mass <= 0.075 and strong_count <= max(2, int(0.04 * w)):
            result[y, xs] = True

    for x in range(w):
        ys = np.flatnonzero(support[:, x])
        if ys.size < max(5, int(0.28 * h)):
            continue
        span = int(ys[-1] - ys[0] + 1)
        mass = float(k[ys, x].sum()) / total
        strong_count = int(np.count_nonzero(strong[ys, x]))
        if span >= int(0.38 * h) and mass <= 0.075 and strong_count <= max(2, int(0.04 * h)):
            result[ys, x] = True

    # Never delete pixels close to the high-confidence core. This protects true
    # intersections and curved trajectories while removing only remote weak tails.
    protected = cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return result & ~protected


def psf_plausibility(kernel: np.ndarray) -> PSFPlausibility:
    k = _normalize(kernel)
    peak = float(k.max())
    if peak <= 0:
        return PSFPlausibility(0.0, 0.0, 1.0, 1.0, 1.0)

    masses = _component_masses(k)
    largest = masses[0] if masses else 0.0
    secondary = float(sum(masses[1:])) if len(masses) > 1 else 0.0
    center, major, anisotropy = _principal_axis(k)

    yy, xx = np.indices(k.shape, dtype=np.float64)
    dx = xx - center[0]
    dy = yy - center[1]
    perpendicular = np.array([-major[1], major[0]], dtype=np.float64)
    distance = np.abs(dx * perpendicular[0] + dy * perpendicular[1])
    band = max(2.0, 0.075 * max(k.shape))
    weak = k < peak * 0.12
    off_axis = float(k[(distance > band) & weak].sum())
    weak_line = float(k[_weak_line_mask(k)].sum())

    return PSFPlausibility(
        largest_component_mass=float(largest),
        secondary_component_mass=float(secondary),
        anisotropy=float(anisotropy),
        off_axis_mass=float(off_axis),
        weak_line_mass=float(weak_line),
    )


def refine_psf_structure(kernel: np.ndarray) -> np.ndarray:
    """Preserve meaningful PSF detail while suppressing weak implausible branches.

    Robust estimation deliberately keeps lower-amplitude support than strict parity
    mode. This pass then removes only two classes of support:

    * tiny disconnected components with negligible probability mass;
    * weak, long row/column spurs far from the high-confidence PSF core.

    Significant secondary connected support is retained, which is important for
    curved/compound trajectories such as the ``toy`` and ``wall`` benchmark cases.
    """
    k = _normalize(kernel)
    peak = float(k.max())
    if peak <= 0:
        return k

    support = k >= peak * 0.015
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        component = labels == label
        mass = float(k[component].sum())
        if mass < 0.012:
            k[component] = 0.0

    # Remove thin low-mass horizontal/vertical tails while retaining strong pixels
    # and the local neighborhood of the main trajectory.
    k[_weak_line_mask(k)] = 0.0

    # For strongly anisotropic single-motion kernels, suppress only very weak mass
    # far away from the core axis. Curved/complex kernels (lower anisotropy) skip
    # this step so secondary trajectory detail is not collapsed.
    diag = psf_plausibility(k)
    if diag.anisotropy >= 5.0:
        center, major, _ = _principal_axis(k)
        yy, xx = np.indices(k.shape, dtype=np.float64)
        perpendicular = np.array([-major[1], major[0]], dtype=np.float64)
        distance = np.abs((xx - center[0]) * perpendicular[0] + (yy - center[1]) * perpendicular[1])
        band = max(2.0, 0.085 * max(k.shape))
        weak = k < float(k.max()) * 0.055
        k[(distance > band) & weak] = 0.0

    return _normalize(k)
