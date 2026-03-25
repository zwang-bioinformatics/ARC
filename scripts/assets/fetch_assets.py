#!/usr/bin/env python3
"""Download ASSET_URLS tarballs to repo root. Toggle ENABLE_*; ``pixi run python -u scripts/assets/fetch_assets.py``.

arc_eval_inputs_core includes only the data/raw_16 files the pipeline needs (see ``make_asset_tarballs_local.py``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Toggle archives to fetch (all on by default).
ENABLE_GRAPH_DATA = True
ENABLE_EVAL_INPUTS_CORE = True
ENABLE_PREDICTIONS_APOLLO = True
ENABLE_PREDICTIONS_ARC = True

ASSET_FLAGS = [
    ("arc_graph_data_casp16.tar.gz", ENABLE_GRAPH_DATA),
    ("arc_eval_inputs_core.tar.gz", ENABLE_EVAL_INPUTS_CORE),
    ("predictions_apollo.tar.gz", ENABLE_PREDICTIONS_APOLLO),
    ("predictions_arc.tar.gz", ENABLE_PREDICTIONS_ARC),
]


def _download(url: str, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"[skip] exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[saved] {out_path}")


def _safe_extract(tf: tarfile.TarFile, dst_root: Path) -> None:
    for member in tf.getmembers():
        member_path = dst_root / member.name
        if not member_path.resolve().as_posix().startswith(dst_root.resolve().as_posix()):
            raise RuntimeError(f"Unsafe tar entry blocked: {member.name}")
    tf.extractall(dst_root)


def _extract(archive_path: Path, repo_root: Path) -> None:
    print(f"[extract] {archive_path} -> {repo_root}")
    with tarfile.open(archive_path, "r:gz") as tf:
        _safe_extract(tf, repo_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        default="downloads/assets_tarballs",
        help="Folder (inside repo) to store tar.gz files before extraction.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload archives even if already present.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    download_dir = repo_root / args.download_dir

    missing = [name for name, _ in ASSET_FLAGS if name not in ASSET_URLS]
    if missing:
        print(f"[error] Missing URL mapping(s): {', '.join(missing)}")
        return 2

    selected = [(name, ASSET_URLS[name]) for name, enabled in ASSET_FLAGS if enabled]
    if not selected:
        print("No assets enabled. Set ENABLE_* = True for the archives you need.")
        return 0

    for name, url in selected:
        archive_path = download_dir / name
        _download(url, archive_path, force=args.force_download)
        _extract(archive_path, repo_root)

    print("\nDone. Assets downloaded and extracted.")
    return 0


# Hosted URLs - update when you upload new tarballs.
ASSET_URLS = {
    "arc_graph_data_casp16.tar.gz": "https://dna.cs.miami.edu/ARC/arc_graph_data_casp16.tar.gz",
    "arc_eval_inputs_core.tar.gz": "https://dna.cs.miami.edu/ARC/arc_eval_inputs_core.tar.gz",
    "predictions_apollo.tar.gz": "https://dna.cs.miami.edu/ARC/predictions_apollo.tar.gz",
    "predictions_arc.tar.gz": "https://dna.cs.miami.edu/ARC/predictions_arc.tar.gz",
}


if __name__ == "__main__":
    sys.exit(main())

