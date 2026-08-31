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
        ROOT / "docs" / "browser-lab.css",
        ROOT / "docs" / "browser-lab.js",
        ROOT / "results" / ".gitkeep",
    ]
    assert all(path.is_file() for path in required)


def test_browser_page_uses_only_local_runtime_assets() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")
    styles = (ROOT / "docs" / "browser-lab.css").read_text(encoding="utf-8")
    lower = page.lower()

    assert 'type="file"' in lower
    assert "analyze &amp; deblur" in lower
    assert "export selected result" in lower
    assert "five deblurring methods" in lower
    assert "before / after" in lower
    assert '<script src="browser-lab.js"></script>' in lower
    assert '<link rel="stylesheet" href="browser-lab.css">' in lower

    for content in (page, script, styles):
        lowered = content.lower()
        assert "https://" not in lowered
        assert "http://" not in lowered
        assert "https://cdn" not in lowered


def test_browser_lab_required_dom_ids_exist() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', page))
    required_match = re.search(r"const REQUIRED_IDS=\[(.*?)\];", script, re.DOTALL)
    assert required_match is not None
    required_ids = set(re.findall(r"'([^']+)'", required_match.group(1)))
    assert required_ids
    assert required_ids <= html_ids


def test_browser_lab_exposes_five_methods_without_parameter_tuning() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")
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
        "function motionConstrainKernel",
        "function rgacRefine",
        "async function restoreFamily",
        "async function renderMethod",
    ]
    for feature in required_features:
        assert feature in script, feature

    for label in [
        "Adaptive Robust Baseline",
        "Motion-Constrained",
        "Annealed PnP",
        "Dual-Extreme",
        "RGAC",
        "Automatic PSF search",
        'id="beforeAfterSlider"',
    ]:
        assert label in page, label

    method_values = set(re.findall(r'name="method" value="([^"]+)"', page))
    assert method_values == {
        "baseline",
        "motion_constrained",
        "annealed_pnp",
        "extreme_channel",
        "rgac",
    }

    # Users may select only the high-level restoration method. Numerical tuning
    # remains automatic, so there are no exposed kernel/gamma/lambda controls.
    assert "<select" not in page.lower()
    assert 'type="number"' not in page.lower()
    assert "manual motion line" not in page.lower()
    assert "draw custom psf" not in page.lower()
    assert "upload psf image" not in page.lower()


def test_browser_before_after_reveal_is_directionally_correct_and_draggable() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")
    styles = (ROOT / "docs" / "browser-lab.css").read_text(encoding="utf-8")

    assert "Drag the center handle" in page
    assert ".viewer .after{clip-path:inset(0 0 0 50%)}" in styles
    assert "pointer-events:auto;cursor:ew-resize;touch-action:none" in styles
    assert "function setSplitFromClientX" in script
    assert "E.splitHandle.addEventListener('pointerdown'" in script
    assert "E.resultImage.style.clipPath=`inset(0 0 0 ${v}%)`" in script


def test_browser_quality_profile_tracks_python_pipeline_more_closely() -> None:
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")

    # The browser remains self-contained, but its quality path should mirror
    # the Python pipeline's stronger blind search and residual-guided consensus.
    assert "estimationMax=Math.min(640" in script
    assert "Math.min(640,plan.estimationMax+120)" in script
    assert "fineIter:5" in script
    assert "Math.SQRT1_2" in script
    assert "Math.min(35,Math.round(35*level))" in script
    assert "function smoothGray" in script
    assert "function localGradientMap" in script
    assert "function localHighpassMap" in script
    assert "projected=restoreRGB" in script


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

    assert int(profiles["7_patch_use.png"]["kernel_size"]) == 85
    assert int(profiles["26.png"]["kernel_size"]) == 69
    assert int(profiles["blurry_7.png"]["kernel_size"]) == 45
    assert profiles["blurry_7.png"]["saturated"] is True
    assert int(profiles["toy.png"]["kernel_size"]) == 101
    assert int(profiles["wall.png"]["kernel_size"]) == 65
