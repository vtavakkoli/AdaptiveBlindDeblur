#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from dark_channel_deblur import DeblurConfig, deblur_image
from dark_channel_deblur.io import read_image, write_image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "real_img2"
RESULTS = ROOT / "results"


def load_image(path: Path, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(a - b)))
    return float("inf") if mse == 0 else float(10 * np.log10(1 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    c1, c2 = 0.01**2, 0.03**2
    scores: list[float] = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        score = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        core = score[5:-5, 5:-5] if min(score.shape) > 10 else score
        scores.append(float(np.mean(core)))
    return float(np.mean(scores))


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    return {"psnr_db": psnr(candidate, reference), "ssim": ssim(candidate, reference)}


def kernel_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, np.float64)
    reference = np.asarray(reference, np.float64)
    candidate /= max(float(candidate.sum()), 1e-12)
    reference /= max(float(reference.sum()), 1e-12)
    return {
        "correlation": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        "l1_distance": float(np.abs(candidate - reference).sum()),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def copy_asset(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def card(title: str, image: str, subtitle: str, badge: str = "") -> str:
    badge_html = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
    return (
        '<article class="card"><div class="card-head"><div>'
        f'<h3>{html.escape(title)}</h3><p>{html.escape(subtitle)}</p></div>{badge_html}</div>'
        f'<img src="{html.escape(image)}" alt="{html.escape(title)}"></article>'
    )


def generate_report(output_dir: Path = RESULTS) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    sources = {
        "input": EXAMPLE / "input_compare.jpg",
        "matlab": EXAMPLE / "matlab_reference" / "result_compare.jpg",
        "previous_full": EXAMPLE / "python" / "full_result_compare.jpg",
        "previous_fast": EXAMPLE / "python" / "fast_result_compare.jpg",
    }
    kernels = {
        "matlab": EXAMPLE / "matlab_reference" / "kernel.png",
        "previous_full": EXAMPLE / "python" / "full_kernel.png",
        "previous_fast": EXAMPLE / "python" / "fast_kernel.png",
    }
    for path in [*sources.values(), *kernels.values(), EXAMPLE / "metrics.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"required benchmark asset is missing: {path}")

    copied = {
        "input": copy_asset(sources["input"], output_dir / "input.jpg"),
        "matlab": copy_asset(sources["matlab"], output_dir / "matlab_reference.jpg"),
        "previous_full": copy_asset(sources["previous_full"], output_dir / "previous_python_full.jpg"),
        "previous_fast": copy_asset(sources["previous_fast"], output_dir / "previous_python_fast.jpg"),
        "matlab_kernel": copy_asset(kernels["matlab"], output_dir / "matlab_kernel.png"),
        "previous_full_kernel": copy_asset(kernels["previous_full"], output_dir / "previous_python_full_kernel.png"),
        "previous_fast_kernel": copy_asset(kernels["previous_fast"], output_dir / "previous_python_fast_kernel.png"),
    }

    config = DeblurConfig(
        kernel_size=25,
        lambda_dark=4e-3,
        lambda_grad=4e-3,
        gamma_correct=1.0,
        xk_iter=5,
        lambda_tv=3e-3,
        lambda_l0=5e-4,
        weight_ring=1.0,
        max_grad_steps=12,
        max_dark_steps=5,
        fft_workers=-1,
    )
    input_float = read_image(sources["input"])
    started = time.perf_counter()
    result, kernel, interim = deblur_image(input_float, config)
    runtime = time.perf_counter() - started

    if result.shape != input_float.shape or not np.isfinite(result).all():
        raise RuntimeError("generated result is invalid")
    if kernel.shape != (25, 25) or not np.isfinite(kernel).all() or float(kernel.sum()) <= 0:
        raise RuntimeError("generated kernel is invalid")

    write_image(output_dir / "new_python_result.png", result)
    write_image(output_dir / "new_python_interim.png", interim)
    write_image(output_dir / "new_python_kernel.png", kernel / max(float(kernel.max()), 1e-12))

    reference = load_image(sources["matlab"])
    image_rows = {
        "blurred_input": image_metrics(load_image(sources["input"]), reference),
        "previous_python_full": image_metrics(load_image(sources["previous_full"]), reference),
        "previous_python_fast": image_metrics(load_image(sources["previous_fast"]), reference),
        "new_python_docker": image_metrics(load_image(output_dir / "new_python_result.png"), reference),
    }
    reference_kernel = load_image(kernels["matlab"], gray=True)
    kernel_rows = {
        "previous_python_full": kernel_metrics(load_image(kernels["previous_full"], gray=True), reference_kernel),
        "previous_python_fast": kernel_metrics(load_image(kernels["previous_fast"], gray=True), reference_kernel),
        "new_python_docker": kernel_metrics(kernel, reference_kernel),
    }
    historical = json.loads((EXAMPLE / "metrics.json").read_text(encoding="utf-8"))

    report = {
        "benchmark": "real_img2 compact CI benchmark",
        "reference": "authors' saved MATLAB output from the supplied CVPR 2016 release",
        "new_method": {
            "name": "Python Docker fast mode",
            "runtime_seconds": runtime,
            "parameters": {
                "kernel_size": config.kernel_size,
                "lambda_dark": config.lambda_dark,
                "lambda_grad": config.lambda_grad,
                "gamma_correct": config.gamma_correct,
                "xk_iter": config.xk_iter,
                "lambda_tv": config.lambda_tv,
                "lambda_l0": config.lambda_l0,
                "max_grad_steps": config.max_grad_steps,
                "max_dark_steps": config.max_dark_steps,
            },
        },
        "preview_agreement_with_matlab": image_rows,
        "preview_kernel_agreement_with_matlab": kernel_rows,
        "historical_full_size_run": historical,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = [
        ("Blurred input", image_rows["blurred_input"], None, "Baseline"),
        ("Previous Python full", image_rows["previous_python_full"], kernel_rows["previous_python_full"], "Previous"),
        ("Previous Python fast", image_rows["previous_python_fast"], kernel_rows["previous_python_fast"], "Previous"),
        ("New Python Docker", image_rows["new_python_docker"], kernel_rows["new_python_docker"], "Current"),
    ]
    metric_rows = "".join(
        f"<tr><td><strong>{html.escape(name)}</strong><span class='pill'>{tag}</span></td>"
        f"<td>{fmt(values['psnr_db'], 2)} dB</td><td>{fmt(values['ssim'])}</td>"
        f"<td>{fmt(kvalues['correlation']) if kvalues else '—'}</td>"
        f"<td>{fmt(kvalues['l1_distance']) if kvalues else '—'}</td></tr>"
        for name, values, kvalues, tag in rows
    )
    cards = "".join([
        card("Blurred input", copied["input"], "Image passed to the Docker test", "Input"),
        card("Original MATLAB", copied["matlab"], "Authors' released CVPR 2016 result", "Reference"),
        card("Previous Python full", copied["previous_full"], "Snapshot from the earlier full Python run", "Previous"),
        card("Previous Python fast", copied["previous_fast"], "Snapshot from the earlier optimized run", "Previous"),
        card("New Python Docker", "new_python_result.png", f"Fresh output generated by docker compose test · {runtime:.2f}s", "Current"),
    ])
    kernel_cards = "".join([
        card("MATLAB kernel", copied["matlab_kernel"], "25×25 reference PSF"),
        card("Previous full kernel", copied["previous_full_kernel"], "Earlier Python estimate"),
        card("Previous fast kernel", copied["previous_fast_kernel"], "Earlier optimized estimate"),
        card("New Docker kernel", "new_python_kernel.png", "Fresh estimate generated in this test"),
    ])
    hist = historical["agreement_with_matlab_reference"]
    hist_kernel = historical["kernel_agreement_with_matlab_reference"]

    report_html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dark Channel Deblur · Docker Comparison Report</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--accent:#335cff;--good:#0f9d75;--shadow:0 12px 36px rgba(15,23,42,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1280px;margin:auto;padding:42px 24px 64px}}
header{{background:linear-gradient(135deg,#111827,#263b73);color:#fff;border-radius:24px;padding:34px 38px;box-shadow:var(--shadow)}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#a9c0ff}}h1{{font-size:34px;line-height:1.15;margin:8px 0}}header p{{margin:0;color:#dbe5ff;max-width:850px}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:22px}}.kpi{{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.16);border-radius:14px;padding:13px 15px}}.kpi b{{display:block;font-size:21px}}.kpi span{{font-size:12px;color:#cdd9f8}}
section{{margin-top:34px}}h2{{font-size:23px;margin:0 0 6px}}.lead{{color:var(--muted);margin:0 0 16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:14px;box-shadow:var(--shadow)}}.card-head{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;margin-bottom:10px}}.card h3{{font-size:15px;margin:0 0 2px}}.card p{{font-size:12px;color:var(--muted);margin:0}}.card img{{display:block;width:100%;height:auto;border-radius:12px;background:#eef2f7}}
.badge,.pill{{display:inline-block;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;background:#e9efff;color:#244bcc;white-space:nowrap}}.pill{{margin-left:8px;background:#eef2f7;color:#64748b}}.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}}table{{border-collapse:collapse;width:100%;min-width:720px}}th,td{{padding:13px 15px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#f8fafc}}tr:last-child td{{border-bottom:0}}
.note{{padding:14px 16px;border-left:4px solid var(--accent);background:#eef3ff;border-radius:10px;color:#31456f}}.historical{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.hist-card{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow)}}.hist-card b{{font-size:25px;color:var(--good)}}code{{background:#eef2f7;border-radius:6px;padding:2px 5px}}footer{{color:var(--muted);font-size:12px;margin-top:30px}}@media(max-width:760px){{main{{padding:20px 14px 40px}}header{{padding:26px 22px}}h1{{font-size:27px}}.kpis,.historical{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Automated Docker validation</div><h1>Dark Channel Deblur comparison report</h1><p>Fresh container output compared with the original MATLAB release and previous Python snapshots. The Docker test creates this report on every run.</p>
<div class="kpis"><div class="kpi"><b>{runtime:.2f}s</b><span>new Docker runtime</span></div><div class="kpi"><b>{fmt(image_rows['new_python_docker']['ssim'])}</b><span>new SSIM vs MATLAB preview</span></div><div class="kpi"><b>{fmt(image_rows['new_python_docker']['psnr_db'],2)} dB</b><span>new PSNR vs MATLAB preview</span></div><div class="kpi"><b>{fmt(kernel_rows['new_python_docker']['correlation'])}</b><span>new kernel correlation</span></div></div></header>
<section><h2>Visual comparison</h2><p class="lead">All images are copied into the result folder, so the HTML stays portable as a CI artifact.</p><div class="grid">{cards}</div></section>
<section><h2>Fresh Docker-run metrics</h2><p class="lead">These metrics use the compact repository test assets and are regenerated inside the container.</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>PSNR vs MATLAB</th><th>SSIM vs MATLAB</th><th>Kernel corr.</th><th>Kernel L1</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
<section><h2>Kernel comparison</h2><p class="lead">Estimated point-spread functions are shown with normalized intensity.</p><div class="grid">{kernel_cards}</div></section>
<section><h2>Previous full-size benchmark</h2><p class="lead">Historical metrics from the earlier 480×360 run are kept separate from the compact CI benchmark.</p><div class="historical"><div class="hist-card"><div class="eyebrow" style="color:#64748b">Previous full</div><b>{hist['python_full']['ssim']:.4f} SSIM</b><p>{hist['python_full']['psnr_db']:.2f} dB PSNR · kernel correlation {hist_kernel['python_full']['correlation']:.4f} · {historical['python_runtime_seconds']['full']:.2f}s</p></div><div class="hist-card"><div class="eyebrow" style="color:#64748b">Previous fast</div><b>{hist['python_fast']['ssim']:.4f} SSIM</b><p>{hist['python_fast']['psnr_db']:.2f} dB PSNR · kernel correlation {hist_kernel['python_fast']['correlation']:.4f} · {historical['python_runtime_seconds']['fast']:.2f}s</p></div></div></section>
<section><div class="note"><strong>Interpretation:</strong> MATLAB is an implementation-fidelity reference, not sharp-image ground truth. The Docker benchmark validates reproducibility and catches regressions; the historical metrics preserve the earlier full-size comparison.</div></section>
<footer>Generated by <code>docker compose run --rm test</code>. Machine-readable metrics: <code>report.json</code>.</footer>
</main></body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path


def main() -> int:
    path = generate_report()
    print(f"Comparison report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
