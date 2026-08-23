from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SATURATED = {
    "26.png",
    "IMG_0650_small_patch.png",
    "IMG_0664_small_patch.png",
    "IMG_4548_small.png",
    "IMG_4561.JPG",
    "blurry_2_small.png",
    "blurry_7.png",
    "my_test_car6.png",
}


def test_profiles_preserve_original_demo_saturation_modes() -> None:
    profiles = json.loads(
        (ROOT / "dataset" / "benchmark_profiles.json").read_text(encoding="utf-8")
    )
    actual = {name for name, profile in profiles.items() if profile.get("saturated")}
    assert actual == EXPECTED_SATURATED
