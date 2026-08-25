from __future__ import annotations

import json
import re
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
        ROOT / "scripts" / "generate_best_report.py",
        ROOT / "scripts" / "generate_matlab_parity_report.py",
        ROOT / "dataset" / "benchmark_profiles.json",
        ROOT / "docs" / "index.html",
        ROOT / "results" / ".gitkeep",
    ]
    assert all(path.is_file() for path in required)


def test_standalone_browser_page_has_no_external_runtime_dependencies() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    lower = page.lower()
    assert "<script" in lower
    assert "<style" in lower
    assert 'type="file"' in lower
    assert "auto deblur" in lower
    assert "export method a" in lower
    assert "export method b" in lower
    assert "<script src=" not in lower
    assert '<link rel="stylesheet"' not in lower
    assert "https://cdn" not in lower


def test_browser_lab_required_dom_ids_exist() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', page))
    required_match = re.search(r"const REQUIRED_IDS=\[(.*?)\];", page, re.DOTALL)
    assert required_match is not None
    required_ids = set(re.findall(r"'([^']+)'", required_match.group(1)))
    assert required_ids
    assert required_ids <= html_ids


def test_browser_lab_is_fully_automatic_and_provides_two_methods() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    required_features = [
        "function analyzeScene",
        "function autoPlan",
        "function autoEstimate",
        "function blindCandidate",
        "function estimatePsfFromGradients",
        "function localMinProjection",
        "function latentStep",
        "function chooseBaseline",
        "function pnpRefine",
        "function extremaRefine",
        "Method A · Robust reconstruction",
        "Method B · Adaptive detail recovery",
        "Automatic PSF search",
    ]
    for feature in required_features:
        assert feature in page, feature

    # User-facing algorithm parameters must stay hidden: the Auto Lab should
    # infer these internally instead of exposing tuning controls.
    assert "<select" not in page.lower()
    assert 'type="number"' not in page.lower()
    assert "manual motion line" not in page.lower()
    assert "draw custom psf" not in page.lower()
    assert "upload psf image" not in page.lower()


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

    # Regression guards for support/mode cases repeatedly exposed by visual audits.
    assert int(profiles["7_patch_use.png"]["kernel_size"]) == 85
    assert int(profiles["26.png"]["kernel_size"]) == 69
    assert int(profiles["blurry_7.png"]["kernel_size"]) == 45
    assert profiles["blurry_7.png"]["saturated"] is True
    assert int(profiles["toy.png"]["kernel_size"]) == 101
    assert int(profiles["wall.png"]["kernel_size"]) == 65
