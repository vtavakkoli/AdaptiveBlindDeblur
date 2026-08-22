from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(Path(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    if bgr.ndim == 2:
        return bgr.astype(np.float32) / 255.0
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image), 0.0, 1.0)
    u8 = np.rint(arr * 255.0).astype(np.uint8)
    if u8.ndim == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    suffix = target.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, u8)
    if not ok:
        raise ValueError(f"OpenCV could not encode {suffix}")
    encoded.tofile(target)
