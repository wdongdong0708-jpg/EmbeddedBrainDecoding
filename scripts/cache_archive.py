#!/usr/bin/env python3
"""Pack and unpack precomputed EEG/MEG feature caches between Colab local NVMe and Google Drive."""

from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path


def pack_cache(cache_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    print(f"[cache_archive] Packing {len(files)} files from {cache_dir} -> {archive_path} ...")
    with tarfile.open(archive_path, "w") as tar:
        tar.add(cache_dir, arcname=".")
    size_mb = archive_path.stat().st_size / (1024.0 * 1024.0)
    print(f"[cache_archive] Done in {time.time() - t0:.1f}s ({size_mb:.1f} MB)")


def unpack_cache(archive_path: Path, cache_dir: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}. Run colab_cpu_cache.ipynb first!")
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[cache_archive] Unpacking {archive_path} -> {cache_dir} ...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=cache_dir)
    print(f"[cache_archive] Unpacked to local NVMe in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "unpack"):
        p = sub.add_parser(name)
        p.add_argument("--cache-dir", type=Path, default=Path("/content/local_cache"))
        p.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "pack":
        pack_cache(args.cache_dir, args.archive)
    else:
        unpack_cache(args.archive, args.cache_dir)


if __name__ == "__main__":
    main()
