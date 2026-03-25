"""Manuscript figure 4 (T1259o): native + three predictor panels via PyMOL, then matplotlib stitch.

Uses ``scripts/manuscript/plot_structure/`` (``generate_pymol_script`` + ``combine_panel_images``).

Requires ``data/casp16_ema_reference_results``, ``data/casp16_targets/T1259o.pdb``,
``data/target_margin_scores.csv``, and the three TS assembly files under
``data/raw_16/T1259o/oligo/``. Pooling pickle from ``raw_pooling_pkl_for_read()`` after
``casp16_eval``; or set ``ARC_FIG4_V4_NO_PICKLE=1`` to skip.

Default CASP models match the T1259o row in ``target_margin_scores.csv``: T1259TS028_2o,
T1259TS294_1o, T1259TS221_1o. Override with ``ARC_FIG4_MODEL_*`` env vars.
Also ``ARC_FIG4_NATIVE_PDB``, ``ARC_FIG4_V4_FILTERED``, ``ARC_FIG4_V4_NO_PICKLE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_dir = os.path.join(os.path.dirname(_script_dir), "eval")
_plot_structure_dir = os.path.join(_script_dir, "plot_structure")
for _p in (_eval_dir, _plot_structure_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from casp16_eval_paths import (  # noqa: E402
    ARC_DATA_DIR,
    CASP16_EMA_REFERENCE_RESULTS_DIR,
    raw_pooling_pkl_for_read,
)

from plots_manuscript_io import manuscript_out_dir  # noqa: E402

FIG4_TARGET = "T1259o"

# Same three models as data/target_margin_scores.csv for T1259o.
_SELECTED = {
    "ARC": "T1259TS028_2o",
    "ModFOLDdock2S": "T1259TS294_1o",
    "MQA_server": "T1259TS221_1o",
}


def _env_model(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _three_models() -> dict[str, str]:
    return {
        "ARC": _env_model("ARC_FIG4_MODEL_ARC", _SELECTED["ARC"]),
        "ModFOLDdock2S": _env_model("ARC_FIG4_MODEL_MODFOLD", _SELECTED["ModFOLDdock2S"]),
        "MQA_server": _env_model("ARC_FIG4_MODEL_MQA", _SELECTED["MQA_server"]),
    }


def _native_pdb_path() -> str:
    env = os.environ.get("ARC_FIG4_NATIVE_PDB", "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.join(ARC_DATA_DIR, "casp16_targets", f"{FIG4_TARGET}.pdb")


def _import_plot_structure():
    import combine_panel_images as c4  # noqa: WPS433
    import generate_pymol_script as g4  # noqa: WPS433

    return g4, c4


def fig4_structural_comparison_t1259o(out_dir: str | None = None, dpi: int = 350) -> str | None:
    """Write fig4.png and fig4.pdf into manuscript_out_dir() or out_dir. dpi unused (combine uses 300)."""
    del dpi

    if not os.path.isdir(_plot_structure_dir):
        print(f"  Fig4 structural: expected {_plot_structure_dir}", flush=True)
        return None

    native_pdb = _native_pdb_path()
    if not os.path.isfile(native_pdb):
        print(
            f"  Fig4 structural: missing native PDB: {native_pdb}",
            flush=True,
        )
        return None

    no_pickle = os.environ.get("ARC_FIG4_V4_NO_PICKLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    pickle_path: str | None = None
    if not no_pickle:
        pickle_path = raw_pooling_pkl_for_read()
        if not os.path.isfile(pickle_path):
            print(
                f"  Fig4 structural: no pickle at {pickle_path}; run eval or ARC_FIG4_V4_NO_PICKLE=1",
                flush=True,
            )
            no_pickle = True
            pickle_path = None

    g4, c4 = _import_plot_structure()
    g4.configure_paths(
        ema_results=os.path.abspath(CASP16_EMA_REFERENCE_RESULTS_DIR),
        raw_pickle=None if no_pickle else os.path.abspath(pickle_path),
        targets_pdb_dir=os.path.join(ARC_DATA_DIR, "casp16_targets"),
    )

    three_override = _three_models()
    print(
        f"  Fig4 structural: models ARC={three_override['ARC']}, "
        f"ModFOLDdock2S={three_override['ModFOLDdock2S']}, "
        f"MQA_server={three_override['MQA_server']}",
        flush=True,
    )

    out = os.path.abspath(out_dir or manuscript_out_dir())
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_fig4_structure_workspace")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    try:
        g4._run_one_target(
            FIG4_TARGET,
            work,
            work,
            three_override=three_override,
            ref_pdb_override=os.path.abspath(native_pdb),
            skip_copy_models=False,
            metrics_override=None,
            raw_pickle_path=None if no_pickle else os.path.abspath(pickle_path),
            no_eval_pickle=no_pickle,
        )
    except SystemExit as e:
        print(f"  Fig4 structural: generate_pymol_script failed: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  Fig4 structural: generate_pymol_script error: {e}", flush=True)
        return None

    pml = os.path.join(work, "manuscript_four_panels.pml")
    if not os.path.isfile(pml):
        print(f"  Fig4 structural: expected PML missing: {pml}", flush=True)
        return None

    pymol_exe = shutil.which("pymol")
    if not pymol_exe:
        print(
            "  Fig4 structural: pymol not on PATH",
            flush=True,
        )
        return None

    env = os.environ.copy()
    env.setdefault("PYMOL_HEADLESS", "1")
    try:
        subprocess.run(
            [pymol_exe, "-cq", pml],
            check=True,
            env=env,
            cwd=work,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "")[-2000:] or (e.stdout or "")[-2000:]
        print(
            f"  Fig4 structural: PyMOL failed (exit {e.returncode}). Last output:\n{tail}",
            flush=True,
        )
        return None

    filtered = os.environ.get("ARC_FIG4_V4_FILTERED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    try:
        c4.main(work, FIG4_TARGET, filtered=filtered)
    except SystemExit as e:
        print(f"  Fig4 structural: combine_panel_images failed: {e}", flush=True)
        return None

    if filtered:
        src_png = os.path.join(work, "manuscript_four_columns_filtered.png")
        src_pdf = os.path.join(work, "manuscript_four_columns_filtered.pdf")
    else:
        src_png = os.path.join(work, "manuscript_four_columns.png")
        src_pdf = os.path.join(work, "manuscript_four_columns.pdf")

    if not os.path.isfile(src_png):
        print(f"  Fig4 structural: missing combined PNG: {src_png}", flush=True)
        return None

    dst_png = os.path.join(out, "fig4.png")
    dst_pdf = os.path.join(out, "fig4.pdf")
    shutil.copy2(src_png, dst_png)
    if os.path.isfile(src_pdf):
        shutil.copy2(src_pdf, dst_pdf)

    print(f"  Fig4 structural: {FIG4_TARGET} -> {dst_png}", flush=True)
    return dst_png


if __name__ == "__main__":
    fig4_structural_comparison_t1259o()
