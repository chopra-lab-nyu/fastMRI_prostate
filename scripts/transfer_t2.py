"""Metadata-driven copier for streaming T2 reconstructions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import pandas as pd
import yaml

ACTIVE_FLAG = ".transfer_active"
STATE_FILE = ".copied_manifest.json"


def human_to_bytes(token: str) -> float:
    if pd.isna(token):
        return float("nan")
    text = str(token).strip().lower()
    match = re.fullmatch(r"([\d.]+)\s*([kmgt]?i?b?)?", text)
    if not match:
        return float("nan")

    value = float(match.group(1))
    suffix = (match.group(2) or "").replace("b", "")
    is_binary = suffix.endswith("i")
    unit = suffix[:-1] if is_binary else suffix
    base = 1024 if is_binary else 1000
    multipliers = {"": 1, "k": base, "m": base**2, "g": base**3, "t": base**4}
    return value * multipliers.get(unit, float("nan"))


def build_t2_manifest(csv_path: Path, min_bytes: int, max_files: int | None = None) -> List[str]:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("Manifest CSV must contain a 'path' column")

    paths = df["path"].astype(str)
    keep = paths.str.contains("AXT2", case=False, na=False) & paths.str.lower().str.endswith(".dat")
    df = df[keep].copy()

    if "size" in df.columns:
        df["size_bytes"] = df["size"].apply(human_to_bytes).astype("float64")
        df = df[df["size_bytes"] >= min_bytes]

    manifest_paths = df["path"].dropna().astype(str).apply(lambda p: p.lstrip("./")).drop_duplicates().tolist()
    if max_files is not None and max_files > 0:
        manifest_paths = manifest_paths[:max_files]
    return manifest_paths


def staging_size_bytes(staging: Path) -> int:
    total = 0

    def _onerror(err: OSError) -> None:
        if isinstance(err, FileNotFoundError):
            return
        raise err

    for root, dirs, files in os.walk(staging, onerror=_onerror):
        dirs[:] = [d for d in dirs if not d.endswith(".lock")]
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).stat().st_size
            except (FileNotFoundError, OSError):
                continue
    return total


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", str(src), str(dest)], check=True)
    else:
        shutil.copy2(src, dest)


def throttle(staging: Path, max_bytes: float, sleep_seconds: int) -> None:
    while staging_size_bytes(staging) >= max_bytes:
        logging.info("Staging at %.1f GB, sleeping %ss", staging_size_bytes(staging) / 1e9, sleep_seconds)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy T2 .dat files into staging for streaming recon")
    parser.add_argument("--config", default="config/streaming_t2.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    transfer_cfg = cfg["transfer"]

    raw_max_files = transfer_cfg.get("max_files")
    max_files = int(raw_max_files) if raw_max_files not in (None, "null", "None") else None
    manifest_paths = build_t2_manifest(
        Path(transfer_cfg["manifest_csv"]),
        int(transfer_cfg["min_bytes"]),
        max_files=max_files,
    )

    source_root = Path(transfer_cfg["source_root"])
    staging_dir = Path(transfer_cfg["staging_dir"])
    staging_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = float(transfer_cfg["max_staging_gb"]) * 1e9
    poll_seconds = int(transfer_cfg["poll_seconds"])

    (staging_dir / ACTIVE_FLAG).touch()

    state_path = staging_dir / STATE_FILE
    copied = set(json.loads(state_path.read_text())["files"]) if state_path.exists() else set()

    for rel in manifest_paths:
        if rel in copied:
            continue

        src = source_root / rel.lstrip("./")
        if not src.exists():
            logging.warning("Missing source %s", src)
            continue

        throttle(staging_dir, max_bytes, poll_seconds)

        dest = staging_dir / src.name
        ready_marker = dest.with_suffix(dest.suffix + ".ready")
        logging.info("Copying %s -> %s", src, dest)
        copy_file(src, dest)
        ready_marker.touch()

        copied.add(rel)
        state_path.write_text(json.dumps({"files": sorted(copied)}, indent=2))

    (staging_dir / ACTIVE_FLAG).unlink(missing_ok=True)
    logging.info("Transfer complete: %d files", len(copied))


if __name__ == "__main__":
    main()
