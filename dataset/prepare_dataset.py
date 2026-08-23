#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "image"

# Official code package linked from Jinshan Pan's dark-channel deblurring page.
OFFICIAL_ZIP = (
    "https://www.dropbox.com/scl/fi/ol4881n658fdpl4oen3ht/"
    "cvpr16_deblurring_code_v1.zip?rlkey=28vtmgs2ukquv6r3j0s17z2vc&dl=1"
)
EXPECTED_COUNT = 23
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)


def _resize_for_ci(path: Path, max_side: int = 192) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode dataset image: {path}")
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(2, round(w * scale)), max(2, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # Normalize the CI working set to lossless PNG while retaining the original
    # stem. This prevents JPEG re-encoding noise from affecting repeatability.
    target = path.with_suffix(".png")
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"could not write CI dataset image: {target}")
    if target != path:
        path.unlink()


def prepare_dataset(*, force: bool = False, max_side: int = 192) -> list[Path]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = _images(DATASET_DIR)
    if not force and len(existing) == EXPECTED_COUNT:
        return existing

    for item in DATASET_DIR.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    with tempfile.TemporaryDirectory(prefix="dark-channel-dataset-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "cvpr16_deblurring_code_v1.zip"
        request = urllib.request.Request(
            OFFICIAL_ZIP,
            headers={"User-Agent": "dark-channel-deblur-ci/0.2"},
        )
        with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)

        if not zipfile.is_zipfile(archive):
            raise RuntimeError("official dataset download is not a valid ZIP archive")

        with zipfile.ZipFile(archive) as zf:
            members = [
                m for m in zf.infolist()
                if not m.is_dir()
                and "/image/" in m.filename.replace("\\", "/")
                and Path(m.filename).suffix.lower() in SUPPORTED
            ]
            if len(members) != EXPECTED_COUNT:
                raise RuntimeError(
                    f"official package contains {len(members)} supported images; expected {EXPECTED_COUNT}"
                )
            for member in members:
                name = Path(member.filename).name
                target = DATASET_DIR / name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    for image_path in _images(DATASET_DIR):
        _resize_for_ci(image_path, max_side=max_side)

    images = _images(DATASET_DIR)
    if len(images) != EXPECTED_COUNT:
        raise RuntimeError(f"prepared {len(images)} images; expected {EXPECTED_COUNT}")
    for path in images:
        if cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
            raise RuntimeError(f"prepared image cannot be decoded: {path}")
    return images


def main() -> int:
    images = prepare_dataset()
    print(f"Prepared {len(images)} official CVPR 2016 images in {DATASET_DIR}")
    for path in images:
        print(f" - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
