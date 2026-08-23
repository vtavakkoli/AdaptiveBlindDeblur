from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SATURATED_CASES = {
    "26.png",
    "IMG_0650_small_patch.png",
    "IMG_0664_small_patch.png",
    "IMG_4548_small.png",
    "IMG_4561.JPG",
    "blurry_2_small.png",
    "blurry_7.png",
    "my_test_car6.png",
}


def test_docker_report_workflow_files_exist() -> None:
    required = [
        ROOT / "scripts" / "run_docker_test.py",
        ROOT / "scripts" / "generate_report.py",
        ROOT / "scripts" / "generate_matlab_parity_report.py",
        ROOT / "dataset" / "benchmark_profiles.json",
        ROOT / "results" / ".gitkeep",
    ]
    assert all(path.is_file() for path in required)


def test_benchmark_profiles_cover_every_source_with_valid_support() -> None:
    dataset = ROOT / "dataset" / "image"
    source_names = {
        path.name
        for path in dataset.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    }
    profiles = json.loads(
        (ROOT / "dataset" / "benchmark_profiles.json").read_text(encoding="utf-8")
    )

    assert len(source_names) == 23
    assert set(profiles) == source_names
    for name, profile in profiles.items():
        size = int(profile["kernel_size"])
        assert size >= 3 and size % 2 == 1, name
        assert float(profile["gamma"]) > 0, name
        assert float(profile["lambda_tv"]) >= 0, name
        assert float(profile["lambda_l0"]) >= 0, name
        assert isinstance(profile["saturated"], bool), name

    configured_saturated = {name for name, p in profiles.items() if p["saturated"]}
    assert configured_saturated == SATURATED_CASES

    # Regression guards for support mismatches exposed by the full visual audit.
    assert int(profiles["7_patch_use.png"]["kernel_size"]) == 85
    assert int(profiles["26.png"]["kernel_size"]) == 69
    assert int(profiles["blurry_7.png"]["kernel_size"]) == 45
