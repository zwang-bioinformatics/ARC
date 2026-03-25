#!/usr/bin/env python3
"""Download ASSET_URLS tarballs to repo root. Toggle ENABLE_*; ``pixi run python -u scripts/assets/fetch_assets.py``.

Archives are ``.tar.zst`` (Zstandard). arc_eval_inputs_core includes only required data/raw_16 files
(see ``make_asset_tarballs_local.py``).

Skips download and extract when the expected tree for that archive already exists under the repo root
unless ``--force-download`` is set.

Extraction creates ``data/``, ``outputs/``, etc. as needed; you do not need to create them first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

import zstandard as zstd

# Toggle archives to fetch (all on by default).
ENABLE_GRAPH_DATA = True
ENABLE_EVAL_INPUTS_CORE = True
ENABLE_PREDICTIONS_APOLLO = True
ENABLE_PREDICTIONS_ARC = True

ASSET_FLAGS = [
    ("arc_graph_data_casp16.tar.zst", ENABLE_GRAPH_DATA),
    ("arc_eval_inputs_core.tar.zst", ENABLE_EVAL_INPUTS_CORE),
    ("predictions_apollo.tar.zst", ENABLE_PREDICTIONS_APOLLO),
    ("predictions_arc.tar.zst", ENABLE_PREDICTIONS_ARC),
]


def _real_dir_nonempty(p: Path) -> bool:
    """Symlinked dirs do not count as installed (extract would write through the link)."""
    return p.is_dir() and not p.is_symlink() and any(p.iterdir())


def _real_file(p: Path) -> bool:
    return p.is_file() and not p.is_symlink()


def _graph_data_present(root: Path) -> bool:
    return _real_dir_nonempty(root / "data" / "CASP16")


def _eval_inputs_core_present(root: Path) -> bool:
    d = root / "data"
    files = [
        d / "casp_model_scores.csv",
        d / "target_margin_scores.csv",
        d / "casp16_approx_target_sizes.json",
        d / "ema_local_scores_with_lddt_added_mdl_contacts.csv",
    ]
    if not all(_real_file(f) for f in files):
        return False
    return all(
        _real_dir_nonempty(p)
        for p in (
            d / "casp16_ema_reference_results",
            d / "casp16_targets",
            d / "raw_16",
        )
    )


def _predictions_apollo_present(root: Path) -> bool:
    base = root / "outputs" / "predictions" / "CASP16"
    if not base.is_dir() or base.is_symlink():
        return False
    for target_dir in base.iterdir():
        if target_dir.is_dir() and (target_dir / "APOLLO").is_dir():
            return True
    return False


def _predictions_arc_present(root: Path) -> bool:
    base = root / "outputs" / "predictions" / "CASP16"
    if not base.is_dir() or base.is_symlink():
        return False
    for target_dir in base.iterdir():
        if not target_dir.is_dir():
            continue
        for method_dir in target_dir.iterdir():
            if method_dir.is_dir() and (
                method_dir.name == "ARC" or method_dir.name.startswith("ARC_")
            ):
                return True
    return False


_ASSET_PRESENT: dict[str, Callable[[Path], bool]] = {
    "arc_graph_data_casp16.tar.zst": _graph_data_present,
    "arc_eval_inputs_core.tar.zst": _eval_inputs_core_present,
    "predictions_apollo.tar.zst": _predictions_apollo_present,
    "predictions_arc.tar.zst": _predictions_arc_present,
}

# If these paths are symlinks, extractall() follows them and never replaces them with real files/dirs.
_STRIP_SYMLINKS_BEFORE_EXTRACT: dict[str, tuple[str, ...]] = {
    "arc_graph_data_casp16.tar.zst": ("data/CASP16",),
    "arc_eval_inputs_core.tar.zst": (
        "data/casp16_ema_reference_results",
        "data/raw_16",
        "data/casp16_targets",
        "data/casp_model_scores.csv",
        "data/casp16_approx_target_sizes.json",
        "data/ema_local_scores_with_lddt_added_mdl_contacts.csv",
        "data/target_margin_scores.csv",
    ),
    "predictions_apollo.tar.zst": ("outputs/predictions/CASP16",),
    "predictions_arc.tar.zst": ("outputs/predictions/CASP16",),
}


def _unlink_pack_symlinks(repo_root: Path, archive_name: str) -> None:
    rels = _STRIP_SYMLINKS_BEFORE_EXTRACT.get(archive_name)
    if not rels:
        return
    for rel in rels:
        p = repo_root / rel
        if p.is_symlink():
            p.unlink()
            print(f"[info] removed symlink before extract: {rel}")


def _already_installed(repo_root: Path, archive_name: str) -> bool:
    fn = _ASSET_PRESENT.get(archive_name)
    return fn(repo_root) if fn else False


def _download(url: str, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"[skip] archive on disk: {out_path}")
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
    kwargs: dict = {}
    if hasattr(tarfile, "data_filter"):
        kwargs["filter"] = tarfile.data_filter
    tf.extractall(dst_root, **kwargs)


def _extract(archive_path: Path, repo_root: Path) -> None:
    """Decompress zstd to a temp seekable .tar; tarfile.extractall() must seek the stream."""
    print(f"[extract] {archive_path} -> {repo_root}")
    dctx = zstd.ZstdDecompressor()
    fd, tmp_name = tempfile.mkstemp(suffix=".tar", prefix="arc_fetch_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(archive_path, "rb") as src, open(tmp_path, "wb") as dst:
            dctx.copy_stream(src, dst)
        with tarfile.open(tmp_path, mode="r:") as tf:
            _safe_extract(tf, repo_root)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        default="downloads/assets_tarballs",
        help="Folder (inside repo) to store .tar.zst files before extraction.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore install checks, redownload archives, and re-extract.",
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
        if not args.force_download and _already_installed(repo_root, name):
            print(f"[skip] already installed: {name}")
            continue

        archive_path = download_dir / name
        _download(url, archive_path, force=args.force_download)
        if not archive_path.is_file():
            print(f"[error] missing archive after download: {archive_path}")
            return 1
        _unlink_pack_symlinks(repo_root, name)
        _extract(archive_path, repo_root)

    print("\nDone. Assets downloaded and extracted.")
    return 0


# Hosted URLs - update when you upload new tarballs.
ASSET_URLS = {
    "arc_graph_data_casp16.tar.zst": "https://dna.cs.miami.edu/ARC/arc_graph_data_casp16.tar.zst",
    "arc_eval_inputs_core.tar.zst": "https://dna.cs.miami.edu/ARC/arc_eval_inputs_core.tar.zst",
    "predictions_apollo.tar.zst": "https://dna.cs.miami.edu/ARC/predictions_apollo.tar.zst",
    "predictions_arc.tar.zst": "https://dna.cs.miami.edu/ARC/predictions_arc.tar.zst",
}


if __name__ == "__main__":
    sys.exit(main())
