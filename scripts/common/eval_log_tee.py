"""Mirror stdout/stderr to the eval log (append), same file as ``casp16_eval.py`` uses.

``casp16_eval.py`` opens ``FULL_PIPELINE_LOG`` in **write** mode for a fresh run.
Manuscript scripts run as **separate processes** (e.g. ``eval-full``); they **append**
here so one continuous log file records eval + figures + tables.

Disable with env ``ARC_NO_EVAL_LOG_TEE=1``.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone


class _TeeOut:
    def __init__(self, stream, f):
        self._stream, self._f = stream, f

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        if data:
            self._f.write(data)
            self._f.flush()

    def flush(self):
        self._stream.flush()
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def append_eval_log_tee(script_label: str) -> None:
    """Append a section to ``FULL_PIPELINE_LOG`` and tee stdout/stderr (console + file)."""
    if os.environ.get("ARC_NO_EVAL_LOG_TEE", "").strip() in ("1", "true", "yes"):
        return

    from casp16_eval_paths import FULL_PIPELINE_LOG, ensure_eval_output_layout

    ensure_eval_output_layout()
    try:
        log_f = open(FULL_PIPELINE_LOG, "a", encoding="utf-8")
    except OSError:
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_f.write(f"\n{'=' * 72}\n")
    log_f.write(f" [{script_label}] started {ts} (append; stdout/stderr tee)\n")
    log_f.write(f"{'=' * 72}\n\n")
    log_f.flush()

    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _TeeOut(orig_out, log_f)
    sys.stderr = _TeeOut(orig_err, log_f)
