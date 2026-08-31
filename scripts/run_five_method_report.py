#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import generate_best_report


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def repair_case_winner_banners() -> None:
    """Ensure each report case has exactly one correctly placed winner banner.

    The base HTML builder predates per-case winner annotations. The five-method
    augmentation inserts those annotations after generation; this final deterministic
    pass removes any provisional banners and places one banner after each case's own
    summary using report.json as the source of truth.
    """
    report_path = RESULTS / "report.json"
    html_path = RESULTS / "report.html"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    text = html_path.read_text(encoding="utf-8")

    text = re.sub(
        r'<div class="case-winner">.*?</div>',
        "",
        text,
        flags=re.DOTALL,
    )

    cases = report.get("images", [])
    index = 0
    pattern = re.compile(r'(<details class="case"[^>]*>.*?</summary>)', re.DOTALL)

    def add_banner(match: re.Match[str]) -> str:
        nonlocal index
        if index >= len(cases):
            return match.group(1)
        case = cases[index]
        index += 1
        winner_key = case["winner"]
        winner_name = report["methods"][winner_key]["name"]
        score = float(case["metrics"][winner_key]["reference_free_score"])
        banner = (
            '<div class="case-winner"><strong>★ Reference-free winner: '
            f"{html.escape(winner_name)}</strong> · score {score:.3f}</div>"
        )
        return match.group(1) + banner

    text = pattern.sub(add_banner, text, count=len(cases))
    if index != len(cases):
        raise RuntimeError(f"placed {index} winner banners for {len(cases)} report cases")
    if text.count('class="case-winner"') != len(cases):
        raise RuntimeError("HTML report does not contain exactly one winner banner per case")

    html_path.write_text(text, encoding="utf-8")


def main() -> int:
    result = generate_best_report.main()
    if result != 0:
        return int(result)
    repair_case_winner_banners()
    print("Per-image winner banners verified: one banner for every report case.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
