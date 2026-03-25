#!/usr/bin/env python3
"""
Combine four PyMOL panel PNGs into one manuscript figure (1 × 4) for ARC fig. 4.

Expects panels from ``generate_pymol_script.py`` + PyMOL (see ``plot_structure/``).

Layout:
  (A) Native Structure | (B) ARC | (C) ModFOLDdock2S | (D) MQA_server

Each model panel shows below the structure image:
  - Quality metrics (Interface ROC AUC, Patch QS, Patch DockQ, Local lDDT, QSBEST, QSGLOB, DOCKQ_AVG)
  - Structural confusion matrix  (model geometry vs native)
  - Evaluator confusion matrix   (evaluator prediction vs native truth)

Reads:
  - outputs/fig4_structure/manuscript/<target>/: pymol_panel_*.png, manuscript_panels_metadata.json
  - data/target_margin_scores.csv (for full metrics: QSBEST, QSGLOB, DOCKQ_AVG)

Writes:
  - manuscript_four_columns.png
  - manuscript_four_columns.pdf

Needs matplotlib + scienceplots. CLI: ``python combine_panel_images.py [--target T]``.
"""

from __future__ import annotations

import argparse
import json
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
ARC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
BASE = ARC_ROOT
from fig4_paths import fig4_scores_csv_path, fig4_structure_work_dir

MARGIN_DIR = fig4_structure_work_dir(ARC_ROOT)
TARGETS_CSV = fig4_scores_csv_path(ARC_ROOT)


def _out_dir_for_target(target: str | None) -> str:
    if target:
        return os.path.join(MARGIN_DIR, "manuscript", target)
    return os.path.join(MARGIN_DIR, "manuscript")

PANELS = ["native", "arc", "modfold", "mqa"]

# Ordered metric labels for model panels (B–D). Keys match CSV prefix per method.
METRIC_LABELS = [
    ("Interface ROC AUC", "interface_roc"),
    ("Patch QS", "gt_patch_qs"),
    ("Patch DockQ", "gt_patch_dockq"),
    ("Local lDDT", "gt_local_lddt"),
    ("QSBEST", "gt_QSBEST"),
    ("QSGLOB", "gt_QSGLOB"),
    ("DOCKQ_AVG", "gt_DOCKQ_AVG"),
]
# Filtered figure: exclude QSGLOB, Local lDDT; no confusion matrices or thr/agree.
METRIC_LABELS_FILTERED = [
    ("Interface ROC AUC", "interface_roc"),
    ("Patch QS", "gt_patch_qs"),
    ("Patch DockQ", "gt_patch_dockq"),
    ("QSBEST", "gt_QSBEST"),
    ("DOCKQ_AVG", "gt_DOCKQ_AVG"),
]
METHOD_CSV_PREFIX = {"native": None, "arc": "ARC", "modfold": "ModFOLDdock2S", "mqa": "MQA_server"}


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------
def load_panel_metadata(meta_path: str) -> list[dict]:
    """Load panel metadata from JSON.  Falls back to defaults if missing."""
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    return [
        {"key": "native",  "title": "(A) Native Structure", "metrics": {}},
        {"key": "arc",     "title": "(B) Top-Model: ARC",   "metrics": {}},
        {"key": "modfold", "title": "(C) Top-Model: ModFOLDdock2S", "metrics": {}},
        {"key": "mqa",     "title": "(D) Top-Model: MQA_server",    "metrics": {}},
    ]


def _format_cm_block(cm: dict, label: str) -> list[str]:
    """Format a 2×2 confusion matrix dict into aligned text lines.

    Example output:
        Structural:
        TP=127  FP= 29
        FN= 15  TN=314
    """
    if not cm:
        return []
    tp = cm.get("TP", 0)
    fp = cm.get("FP", 0)
    fn = cm.get("FN", 0)
    tn = cm.get("TN", 0)
    w = max(len(str(tp)), len(str(fp)), len(str(fn)), len(str(tn)))
    lines = [
        f"{label}:",
        f"  TP={tp:>{w}}  FP={fp:>{w}}",
        f"  FN={fn:>{w}}  TN={tn:>{w}}",
    ]
    return lines


# ---------------------------------------------------------------------------
# Augment panel metrics from target_margin_scores.csv
# ---------------------------------------------------------------------------
def _metrics_for_panel(panel: dict, target: str | None, csv_row, metric_labels=None) -> dict[str, float]:
    """Build full metrics dict for this panel: metadata + CSV (ordered)."""
    if metric_labels is None:
        metric_labels = METRIC_LABELS
    out: dict[str, float] = {}
    key = panel.get("key", "")
    prefix = METHOD_CSV_PREFIX.get(key)
    if prefix and csv_row is not None and target:
        for label, col_suffix in metric_labels:
            if col_suffix == "interface_roc":
                csv_key = f"{prefix}_interface_roc"
            else:
                csv_key = f"{prefix}_{col_suffix}"
            if csv_key in csv_row.index and str(csv_row.get(csv_key, "")) not in ("", "nan", "None"):
                try:
                    out[label] = float(csv_row[csv_key])
                except (TypeError, ValueError):
                    pass
    if not out:
        out = dict(panel.get("metrics", {}))
    return out


def _best_panels_per_metric(meta: list[dict], target: str | None, csv_row, metric_labels: list) -> dict[str, set[str]]:
    """For each metric label, return set of panel keys (arc, modfold, mqa) that have the highest value."""
    import math
    best: dict[str, set[str]] = {}
    evaluator_keys = ["arc", "modfold", "mqa"]
    for label, col_suffix in metric_labels:
        values = {}
        for panel in meta:
            key = panel.get("key", "")
            if key not in evaluator_keys:
                continue
            m = _metrics_for_panel(panel, target, csv_row, metric_labels)
            if label in m and not math.isnan(m[label]):
                values[key] = m[label]
        if not values:
            best[label] = set()
            continue
        max_val = max(values.values())
        best[label] = {k for k, v in values.items() if v == max_val}
    return best


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def main(out_dir: str, target: str | None, filtered: bool = False):
    global OUT_DIR, OUT_PNG, OUT_PDF
    OUT_DIR = out_dir
    if filtered:
        OUT_PNG = os.path.join(OUT_DIR, "manuscript_four_columns_filtered.png")
        OUT_PDF = os.path.join(OUT_DIR, "manuscript_four_columns_filtered.pdf")
        metric_labels = METRIC_LABELS_FILTERED
    else:
        OUT_PNG = os.path.join(OUT_DIR, "manuscript_four_columns.png")
        OUT_PDF = os.path.join(OUT_DIR, "manuscript_four_columns.pdf")
        metric_labels = METRIC_LABELS

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except ImportError:
        raise SystemExit(
            "scienceplots is required. Install with: pip install scienceplots"
        )

    import matplotlib.image as mpimg

    def load_img(path: str) -> np.ndarray:
        return mpimg.imread(path)

    def crop_white_border(img: np.ndarray, margin: int = 4) -> np.ndarray:
        """Crop image to content bounding box, trimming excess white border."""
        if img.ndim == 3:
            gray = np.max(img[..., :3], axis=2)
        else:
            gray = img
        white_thresh = 240.0 if np.nanmax(gray) > 1.01 else 0.95
        non_white = gray < white_thresh
        if not np.any(non_white):
            return img
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - margin)
        rmax = min(img.shape[0], rmax + 1 + margin)
        cmin = max(0, cmin - margin)
        cmax = min(img.shape[1], cmax + 1 + margin)
        return img[rmin:rmax, cmin:cmax]

    paths = [os.path.join(OUT_DIR, f"pymol_panel_{p}.png") for p in PANELS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing panel images — run PyMOL first.\nMissing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    meta_path = os.path.join(OUT_DIR, "manuscript_panels_metadata.json")
    meta = load_panel_metadata(meta_path)
    if not target and meta and meta[0].get("target"):
        target = meta[0]["target"]
    csv_row = None
    csv_path = TARGETS_CSV
    if target and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[df["target"] == target]
        if not row.empty:
            csv_row = row.iloc[0]

    # Light crop: trim a little white border only (large margin = minimal zoom)
    images = [crop_white_border(load_img(p), margin=48) for p in paths]

    # Native panel only: add padding on bottom, left, and right (not top) for breathing room
    pad_left = images[0].shape[1] // 12
    pad_right = images[0].shape[1] // 12
    pad_bottom = images[0].shape[0] // 10
    pad_top = 0
    if images[0].ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    white = 1.0 if np.issubdtype(images[0].dtype, np.floating) else 255
    images[0] = np.pad(images[0], pad_width, mode="constant", constant_values=white)

    # Native panel only: scale down structure and center so it appears smaller (no tail like B-D)
    try:
        from scipy.ndimage import zoom as ndi_zoom
        NATIVE_SCALE = 0.82  # show native at 82% size so main body matches apparent size of B-D
        img0 = images[0]
        h, w = img0.shape[0], img0.shape[1]
        if img0.ndim == 3:
            zoom_factors = (NATIVE_SCALE, NATIVE_SCALE, 1.0)
        else:
            zoom_factors = (NATIVE_SCALE, NATIVE_SCALE)
        scaled = ndi_zoom(img0.astype(np.float64), zoom_factors, order=1)
        if img0.ndim == 3:
            sh, sw = scaled.shape[0], scaled.shape[1]
        else:
            sh, sw = scaled.shape[0], scaled.shape[1]
        pad_h, pad_w = (h - sh) // 2, (w - sw) // 2
        canvas = np.full((h, w) if img0.ndim == 2 else (h, w, img0.shape[2]), white, dtype=img0.dtype)
        if np.issubdtype(img0.dtype, np.floating):
            scaled = np.clip(scaled, 0, 1).astype(img0.dtype)
        else:
            scaled = np.clip(scaled, 0, 255).astype(img0.dtype)
        canvas[pad_h : pad_h + sh, pad_w : pad_w + sw] = scaled
        images[0] = canvas
    except Exception:
        pass  # skip scale-down if scipy missing or error

    # Which evaluator panel(s) have the best value for each metric (for bold)
    best_for_metric = _best_panels_per_metric(meta, target, csv_row, metric_labels)

    # Two-column figure*: width fits \textwidth when scaled in LaTeX
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
    LINE_HEIGHT = 0.048
    # Value column: right-aligned at 0.98 so "Interface ROC AUC" and score never overlap
    VALUE_X_RIGHT = 0.98

    for ax, img, panel in zip(axes, images, meta):
        ax.imshow(img, aspect="equal")
        ax.set_title(panel["title"], fontsize=12, pad=4)
        ax.set_axis_off()

        panel_metrics = _metrics_for_panel(panel, target, csv_row, metric_labels)
        panel_key = panel.get("key", "")
        evaluator_keys = ("arc", "modfold", "mqa")

        # Build lines: (label_part, value_part, bold) for metrics (value_part None = single line)
        # Label left-aligned, value right-aligned so numbers line up across panels when some are bold.
        lines_display: list[tuple[str, str | None, bool]] = []
        for label, _ in metric_labels:
            if label not in panel_metrics:
                continue
            value = panel_metrics[label]
            value_str = f"{value:.3f}"
            is_best = (
                panel_key in evaluator_keys
                and panel_key in best_for_metric.get(label, set())
            )
            lines_display.append((f"{label}: ", value_str, is_best))

        if not filtered:
            # Structural confusion matrix
            scm = panel.get("confusion_matrix")
            if scm:
                if lines_display:
                    lines_display.append(("", None, False))
                for ln in _format_cm_block(scm, "Structural"):
                    lines_display.append((ln, None, False))

            # Evaluator confusion matrix and thr/agree
            ecm = panel.get("eval_confusion_matrix")
            if ecm:
                lines_display.append(("", None, False))
                for ln in _format_cm_block(ecm, "Evaluator"):
                    lines_display.append((ln, None, False))
                thresh = panel.get("eval_threshold")
                rate = panel.get("eval_agreement_rate")
                detail = []
                if thresh is not None:
                    detail.append(f"thr={thresh:.3f}")
                if rate is not None:
                    detail.append(f"agree={rate:.1%}")
                if detail:
                    lines_display.append((f"  ({', '.join(detail)})", None, False))

        if lines_display:
            text_kw = dict(
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                family="monospace",
                clip_on=False,
            )
            has_value_lines = any(v is not None for _, v, _ in lines_display)
            if has_value_lines:
                for i, (label_part, value_part, bold) in enumerate(lines_display):
                    y = -0.03 - i * LINE_HEIGHT
                    ax.text(0.02, y, label_part, ha="left", **text_kw)
                    if value_part is not None:
                        ax.text(
                            VALUE_X_RIGHT, y, value_part,
                            ha="right",
                            fontweight="bold" if bold else "normal",
                            **text_kw,
                        )
            else:
                ax.text(
                    0.5, -0.03,
                    "\n".join(l for l, _, _ in lines_display),
                    ha="center",
                    **text_kw,
                    linespacing=1.3,
                )

    # ── Legend in panel A blank space ──────────────────────────────────
    ax0 = axes[0]
    leg_kw = dict(transform=ax0.transAxes, fontsize=8, va="center",
                  clip_on=False)
    y0 = -0.04
    rh = 0.050          # row height
    ms = 5.5            # marker size

    # PyMOL chain colours (red, marine, forest)
    tp_colors = ["#FF0000", "#007FFF", "#339933"]

    # Row 1: TP — three chain-colour swatches
    for j, c in enumerate(tp_colors):
        ax0.plot(0.04 + j * 0.06, y0, marker="s", color=c, alpha=1.0,
                 markersize=ms, transform=ax0.transAxes, clip_on=False)
    ax0.text(0.04 + len(tp_colors) * 0.06, y0,
             "TP (true interface residue on chain: A, B, C)",
             ha="left", **leg_kw)

    # Row 2: FP
    y1 = y0 - rh
    ax0.plot(0.04, y1, marker="s", color="#FF8C00", alpha=1.0,
             markersize=ms, transform=ax0.transAxes, clip_on=False)
    ax0.text(0.10, y1, "FP (false interface residue)", ha="left", **leg_kw)

    # Row 3: FN (circle = sphere representation)
    y2 = y0 - 2 * rh
    ax0.plot(0.04, y2, marker="o", color="#FF8C00", alpha=1.0,
             markersize=ms, transform=ax0.transAxes, clip_on=False)
    ax0.text(0.10, y2, "FN (missed interface residue)", ha="left", **leg_kw)

    # Row 4: TN
    y3 = y0 - 3 * rh
    ax0.plot(0.04, y3, marker="s", color="#999999", alpha=0.55,
             markersize=ms, transform=ax0.transAxes, clip_on=False)
    ax0.text(0.10, y3, "TN (true non-interface residue)", ha="left", **leg_kw)

    # Row 5: Opacity — color-agnostic, two swatches of same neutral tone
    y4 = y0 - 4 * rh
    _oc = "#555555"  # neutral dark grey
    ax0.plot(0.04, y4, marker="s", color=_oc, alpha=1.0,
             markersize=ms, transform=ax0.transAxes, clip_on=False)
    ax0.text(0.10, y4, "Opaque = eval. agrees", ha="left", **leg_kw)
    ax0.plot(0.54, y4, marker="s", color=_oc, alpha=0.30,
             markersize=ms, markeredgecolor=_oc, markeredgewidth=0.5,
             transform=ax0.transAxes, clip_on=False)
    ax0.text(0.60, y4, "Faded = eval. disagrees", ha="left", **leg_kw)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(bottom=0.14, left=0.02, right=0.98, top=0.90, wspace=0.06)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Combine fig. 4 PyMOL panels into one PNG/PDF (ARC repo).")
    ap.add_argument(
        "--target",
        metavar="T",
        help="Use outputs/fig4_structure/manuscript/<target>/ (default: outputs/.../manuscript/).",
    )
    ap.add_argument(
        "--all-targets",
        action="store_true",
        help="Combine for every target in target_margin_scores.csv that has panel dirs under manuscript/.",
    )
    ap.add_argument("--filtered", action="store_true", help="Output filtered figure: metrics only (no QSGLOB/Local lDDT, no confusion matrices or thr/agree). Writes manuscript_four_columns_filtered.png/.pdf.")
    args = ap.parse_args()
    if args.all_targets:
        import pandas as pd
        csv_path = TARGETS_CSV
        if not os.path.exists(csv_path):
            raise SystemExit(
                "target_margin_scores.csv not found under data/. Run build_targets_scores_csv.py first."
            )
        df = pd.read_csv(csv_path)
        targets = [str(t).strip() for t in df["target"].unique() if t and str(t).strip() != "(none)"]
        for t in targets:
            out_dir = os.path.join(MARGIN_DIR, "manuscript", t)
            if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "pymol_panel_native.png")):
                main(out_dir, t, filtered=args.filtered)
                print("Combined: %s%s" % (t, " (filtered)" if args.filtered else ""))
            else:
                print("Skip (missing panels): %s" % t, file=__import__("sys").stderr)
        print("Done: %d targets" % len(targets))
    else:
        out_dir = _out_dir_for_target(args.target)
        main(out_dir, args.target, filtered=args.filtered)
