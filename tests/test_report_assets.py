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
    """Every shipped runtime asset is local and present in the Pages directory."""
    from html.parser import HTMLParser

    class Assets(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.urls: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            if tag == "script" and values.get("src"):
                self.urls.append(values["src"])
            if tag == "link" and values.get("rel") == "stylesheet":
                self.urls.append(values["href"])

    parser = Assets()
    parser.feed((ROOT / "docs" / "index.html").read_text(encoding="utf-8"))
    assert set(parser.urls) == {"browser-lab.css", "browser-lab.js", "deblur-core.js"}
    for asset in [*parser.urls, "deblur-worker.js"]:
        content = (ROOT / "docs" / asset).read_text(encoding="utf-8")
        assert "https://" not in content and "http://" not in content


def test_browser_lab_required_dom_ids_exist() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")
    html_ids = re.findall(r'id="([A-Za-z0-9_-]+)"', page)
    assert len(html_ids) == len(set(html_ids)), "Duplicate element IDs break control binding"
    required_match = re.search(r"const ids\s*=\s*\[(.*?)\];", script, re.DOTALL)
    assert required_match is not None
    required_ids = set(re.findall(r"[\"']([A-Za-z0-9_-]+)[\"']", required_match.group(1)))
    assert required_ids and required_ids <= set(html_ids)


def test_browser_preserves_five_methods_and_exposes_resolution_and_blur_controls() -> None:
    from html.parser import HTMLParser

    class Controls(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.methods: set[str] = set()
            self.labels: set[str] = set()
            self.ids: set[str] = set()

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            if tag == "input" and values.get("name") == "method":
                self.methods.add(values["value"])
            if tag == "label" and values.get("for"):
                self.labels.add(values["for"])
            if values.get("id"):
                self.ids.add(values["id"])

    controls = Controls()
    controls.feed((ROOT / "docs" / "index.html").read_text(encoding="utf-8"))
    assert controls.methods == {"baseline", "motion_constrained", "annealed_pnp", "extreme_channel", "rgac"}
    assert {"resolutionSelect", "modelSelect", "motionLength", "motionAngle", "defocusRadius",
            "cancelBtn", "zoomSelect", "beforeAfterSlider", "exportBtn", "reportBtn"} <= controls.ids
    assert "fileInput" in controls.labels


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
