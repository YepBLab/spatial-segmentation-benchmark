#!/usr/bin/env python3
"""Build a data-driven HTML report without study- or model-specific claims."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from segbench.io import load_yaml


CORE_METRICS = [
    ("mSA_50_95", "mSA@[0.50:0.95]"),
    ("SA50", "SA50"),
    ("SA75", "SA75"),
    ("precision_iou50", "Precision@0.50"),
    ("recall_iou50", "Recall@0.50"),
    ("f1_iou50", "F1@0.50"),
    ("matched_iou_mean", "Matched IoU"),
    ("matched_dice_mean", "Matched Dice"),
    ("DQ", "DQ"),
    ("SQ", "SQ"),
    ("PQ", "PQ"),
    ("AJI_plus", "AJI+"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def metric_cell(rows: pd.DataFrame, metric: str) -> str:
    row = rows.loc[rows["metric"] == metric]
    if row.empty:
        return "—"
    item = row.iloc[0]
    estimate = float(item["estimate"])
    low = float(item["ci_low"])
    high = float(item["ci_high"])
    if not np.isfinite(estimate):
        return "—"
    if np.isfinite(low) and np.isfinite(high):
        return f"{estimate:.3f}<br><small>[{low:.3f}, {high:.3f}]</small>"
    return f"{estimate:.3f}"


def accuracy_table(macro: pd.DataFrame, model_order: list[str]) -> str:
    labels = {key: label for key, label in CORE_METRICS}
    header = "".join(f"<th>{html.escape(labels[key])}</th>" for key, _ in CORE_METRICS)
    body: list[str] = []
    for model_key in model_order:
        subset = macro.loc[macro["model_key"] == model_key]
        if subset.empty:
            continue
        display = str(subset.iloc[0]["display"])
        cells = "".join(f"<td>{metric_cell(subset, key)}</td>" for key, _ in CORE_METRICS)
        body.append(f"<tr><th>{html.escape(display)}</th>{cells}</tr>")
    return f"<table><thead><tr><th>Model</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def boundary_table(boundary: pd.DataFrame, model_order: list[str]) -> str:
    if boundary.empty:
        return "<p>Boundary sensitivity results were not available.</p>"
    chunks: list[str] = []
    for tolerance in sorted(boundary["tolerance_um"].dropna().unique()):
        rows = boundary.loc[boundary["tolerance_um"] == tolerance]
        body: list[str] = []
        for model_key in model_order:
            subset = rows.loc[rows["model_key"] == model_key]
            if subset.empty:
                continue
            display = str(subset.iloc[0]["display"])
            body.append(
                "<tr>"
                f"<th>{html.escape(display)}</th>"
                f"<td>{metric_cell(subset, 'boundary_precision')}</td>"
                f"<td>{metric_cell(subset, 'boundary_recall')}</td>"
                f"<td>{metric_cell(subset, 'boundary_f1')}</td>"
                f"<td>{metric_cell(subset, 'nsd')}</td>"
                "</tr>"
            )
        chunks.append(
            f"<h3>Tolerance: {float(tolerance):g} µm</h3>"
            "<table><thead><tr><th>Model</th><th>Boundary precision</th>"
            "<th>Boundary recall</th><th>Boundary F1</th><th>NSD</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )
    return "".join(chunks)


def utility_table(utility: pd.DataFrame, model_order: list[str]) -> str:
    if utility.empty:
        return "<p>Global/medullary utility results were not supplied.</p>"
    columns = [
        ("cell_count", "Cells"),
        ("cells_per_mm2", "Cells/mm²"),
        ("transcripts_per_cell_median", "Median transcripts/cell"),
        ("genes_per_cell_median", "Median genes/cell"),
    ]
    available = [(key, label) for key, label in columns if key in utility.columns]
    rows: list[str] = []
    for model_key in model_order:
        for scope in ("global", "medullary", "outside_medullary"):
            subset = utility.loc[
                (utility["model_key"] == model_key) & (utility["scope"] == scope)
            ]
            if subset.empty:
                continue
            item = subset.iloc[0]
            values = "".join(
                f"<td>{float(item[key]):,.1f}</td>" if pd.notna(item[key]) else "<td>—</td>"
                for key, _ in available
            )
            rows.append(
                f"<tr><th>{html.escape(str(item['display']))}</th>"
                f"<td>{html.escape(scope)}</td>{values}</tr>"
            )
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in available)
    return (
        "<table><thead><tr><th>Model</th><th>Scope</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def figure_block(figures_dir: Path, filename: str, title: str, caption: str) -> str:
    path = figures_dir / filename
    if not path.exists():
        return ""
    return (
        "<figure>"
        f'<img src="../figures/{html.escape(filename)}" alt="{html.escape(title)}">'
        f"<figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption>"
        "</figure>"
    )


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics_dir = project / "metrics"
    figures_dir = project / "figures"
    report_dir = project / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    macro = pd.read_csv(metrics_dir / "manual_macro_summary.csv")
    boundary_path = metrics_dir / "boundary_sensitivity_macro.csv"
    utility_path = metrics_dir / "global_medullary_utility.csv"
    boundary = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
    utility = pd.read_csv(utility_path) if utility_path.exists() else pd.DataFrame()
    model_order = list(registry["models"])

    msa = macro.loc[macro["metric"] == "mSA_50_95"].sort_values("estimate", ascending=False)
    leader = None if msa.empty else msa.iloc[0]
    title = str(config.get("report_title", "Spatial instance-segmentation benchmark"))
    design = str(config.get("evaluation_design", "unspecified"))
    n_regions = int(macro["n_regions"].max()) if "n_regions" in macro and not macro.empty else 0
    leader_text = (
        "No mSA result was available."
        if leader is None
        else f"{html.escape(str(leader['display']))} has the highest observed macro mSA "
        f"({float(leader['estimate']):.3f}); interpret the ranking in light of the declared evaluation design."
    )

    figures = "".join(
        [
            figure_block(figures_dir, "01_manual_accuracy_scorecard.png", "Manual accuracy scorecard", "Macro mean and ROI-bootstrap confidence intervals."),
            figure_block(figures_dir, "02_sa_threshold_curve.png", "SA threshold curve", "Instance accuracy as the IoU requirement becomes stricter."),
            figure_block(figures_dir, "03_detection_iou50.png", "Detection precision and recall", "IoU 0.50 detection trade-off."),
            figure_block(figures_dir, "04_boundary_tolerance_sensitivity.png", "Boundary sensitivity", "Boundary scores across tolerances expressed in micrometres."),
            figure_block(figures_dir, "05_pq_decomposition.png", "PQ decomposition", "PQ shown together with detection quality (DQ) and segmentation quality (SQ)."),
            figure_block(figures_dir, "06_error_spectrum.png", "Error spectrum", "Miss, spurious, split, merge, poor-overlap and complex events."),
            figure_block(figures_dir, "07_size_density_strata.png", "Size and density strata", "Recall and overlap quality by reference-cell size and local density."),
            figure_block(figures_dir, "08a_global_utility.png", "Global utility", "Whole-tissue cell and transcript-derived summaries without manual ground truth."),
            figure_block(figures_dir, "08b_medullary_utility.png", "Medullary utility", "Medullary-region cell and transcript-derived summaries without manual ground truth."),
            figure_block(figures_dir, "08c_scope_utility_contrast.png", "Scope utility contrast", "Medullary-versus-global descriptive contrasts."),
            figure_block(figures_dir, "09_representative_failures.png", "Representative failure cases", "Image/mask panels selected by predeclared error categories."),
        ]
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#0b1118;--panel:#121b26;--ink:#edf4fb;--muted:#9fb0c2;--line:#2a3b4f;--accent:#5cc8ff;--warn:#ffd166}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:36px}} h1{{font-size:34px;margin-bottom:4px}} h2{{margin-top:40px;color:var(--accent)}}
.meta,.note{{color:var(--muted)}} .callout{{background:var(--panel);border-left:4px solid var(--warn);padding:14px 18px;margin:20px 0}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;background:var(--panel);margin:14px 0 26px}}
th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
figure{{background:var(--panel);padding:16px;margin:24px 0}} img{{display:block;width:100%;height:auto}} figcaption{{color:var(--muted);margin-top:10px}}
code{{color:#a7e3ff}} small{{color:var(--muted)}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · {n_regions} reference regions · evaluation design: {html.escape(design)}</p>
<div class="callout"><strong>Observed result:</strong> {leader_text}</div>
<p>This report separates manual-reference accuracy, matched-boundary quality, instance error types,
stratified performance, and whole-tissue utility. Global/medullary cell and transcript summaries are
descriptive utility measures, not substitutes for manual-reference accuracy.</p>

<h2>Manual-reference accuracy</h2><div class="table-wrap">{accuracy_table(macro, model_order)}</div>
<p class="note">Each cell is the ROI-macro estimate with a percentile cluster-bootstrap interval.
mSA is the mean SA across IoU thresholds 0.50–0.95. Precision, recall and F1 use IoU 0.50.</p>

<h2>Boundary agreement</h2><div class="table-wrap">{boundary_table(boundary, model_order)}</div>
<p class="note">Boundary metrics are computed only for IoU-0.50 matched pairs. A biological primary
tolerance should be chosen from repeated-annotation disagreement; until then, all tolerances are sensitivity analysis.</p>

<h2>Global and medullary utility</h2><div class="table-wrap">{utility_table(utility, model_order)}</div>
<p class="note">These summaries describe output scale and transcript assignment. They do not have dense
manual ground truth and therefore should not be called segmentation accuracy.</p>

<h2>Figures</h2>{figures}

<h2>Interpretation safeguards</h2>
<ul><li>Use mSA as the primary instance-accuracy endpoint; report SA50, SA75 and the full curve.</li>
<li>Interpret DQ and SQ with PQ: DQ is detection correspondence, SQ is matched-pair overlap.</li>
<li>Use AJI+ as a pathology-literature bridge, not as the sole conclusion.</li>
<li>Review misses, spurious objects, splits and merges together with representative image panels.</li>
<li>Do not claim generalization when training and evaluation use the same reference regions.</li></ul>
</main></body></html>"""

    report_path = report_dir / "segmentation_benchmark_report.html"
    report_path.write_text(document, encoding="utf-8")
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "evaluation_design": design,
        "reference_regions": n_regions,
        "models": model_order,
        "top_observed_msa_model": None if leader is None else str(leader["model_key"]),
        "top_observed_msa": None if leader is None else float(leader["estimate"]),
        "report": str(report_path),
    }
    (report_dir / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
