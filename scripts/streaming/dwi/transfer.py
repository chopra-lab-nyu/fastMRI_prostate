"""Metadata-driven copier for streaming DWI reconstructions."""
from __future__ import annotations

import argparse
import os
import json
import logging
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yaml

ACTIVE_FLAG = ".transfer_active"
STATE_FILE = ".copied_manifest.json"
PRESCAN_MAX_BYTES = 100e6
MAIN_MIN_BYTES = 1e9
PAIR_TOLERANCE = pd.Timedelta("5min")


def human_to_bytes(token: str) -> float:
    """Convert human-readable sizes like '3.6G' or '2GiB' to bytes."""
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


def parse_dat_filename(path: str) -> dict[str, object]:
    """Extract scanner metadata encoded in Siemens .dat filenames."""
    info: dict[str, object] = {}
    parts = Path(path).stem.split("#")
    for part in parts:
        if part.startswith("S"):
            info["scanner_site"] = part[1:]
        elif part.startswith("F"):
            info["F_field"] = part[1:]
        elif part.startswith("M"):
            info["M_field"] = part[1:]
        elif re.fullmatch(r"D\d{6}", part):
            info["date"] = datetime.strptime(part[1:], "%d%m%y").date().isoformat()
        elif re.fullmatch(r"T\d{6}", part):
            info["time"] = datetime.strptime(part[1:], "%H%M%S").time().isoformat()
        elif not part.startswith(("Hersh_VidaProstateDiffusion", "PidaProstateDiffusion")):
            info["modality"] = part
        
    info['fname_without_site'] = '#'.join([part for part in parts if not part.startswith("S")])

    if "date" in info and "time" in info:
        try:
            info["datetime"] = datetime.fromisoformat(f"{info['date']}T{info['time']}")
        except ValueError:
            info["datetime"] = pd.NaT
    else:
        info["datetime"] = pd.NaT
    return info


def build_main_manifest(csv_path: Path, min_bytes: int, max_files: int | None = None) -> List[str]:
    df_all = pd.read_csv(csv_path)
    if "path" not in df_all.columns or "size" not in df_all.columns:
        raise ValueError("Manifest CSV must contain 'path' and 'size' columns")

    path_strings = df_all["path"].astype(str)
    keep_axial = path_strings.str.contains("AX", case=False, na=False)
    skip_dl = path_strings.str.contains("DL_DIFFUSION", case=False, na=False)
    df_all = df_all[keep_axial & ~skip_dl].copy()

    parsed = df_all["path"].apply(parse_dat_filename).apply(pd.Series)
    df_all = pd.concat([df_all, parsed], axis=1)
    df_all["size_bytes"] = df_all["size"].apply(human_to_bytes).astype("float64")
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    df_all["scanner_site"] = df_all["scanner_site"].astype(str)
    df_all["F_field"] = pd.to_numeric(df_all["F_field"], errors="coerce")
    df_all["M_field"] = pd.to_numeric(df_all["M_field"], errors="coerce")

    df_main = df_all[df_all["size_bytes"] >= max(MAIN_MIN_BYTES, min_bytes)].copy()
    df_pres = df_all[df_all["size_bytes"] <= PRESCAN_MAX_BYTES].copy()

    df_main = (
        df_main.dropna(subset=["scanner_site", "datetime"])
        .sort_values(["datetime", "scanner_site"])
        .rename(columns=lambda c: f"{c}_main" if c not in {"scanner_site", "datetime"} else c)
    )
    df_pres = (
        df_pres.dropna(subset=["scanner_site", "datetime"])
        .sort_values(["datetime", "scanner_site"])
        .rename(columns=lambda c: f"{c}_pre" if c not in {"scanner_site", "datetime"} else c)
    )
    df_pres["datetime_pre"] = df_pres["datetime"]

    if df_main.empty:
        logging.warning("No main scans found in manifest %s", csv_path)
        return []

    pairs = pd.merge_asof(
        df_main,
        df_pres,
        on="datetime",
        by="scanner_site",
        direction="nearest",
        tolerance=PAIR_TOLERANCE,
        suffixes=("_main", "_pre"),
    )

    pairs = pairs.rename(columns={"datetime": "datetime_main"})
    if "F_field_main" in pairs and "F_field_pre" in pairs:
        pairs["dF"] = (pairs["F_field_main"] - pairs["F_field_pre"]).abs()
    if "M_field_main" in pairs and "M_field_pre" in pairs:
        pairs["dM"] = (pairs["M_field_main"] - pairs["M_field_pre"]).abs()
    if "datetime_pre" in pairs:
        pairs["delta_minutes"] = (
            (pairs["datetime_main"] - pairs["datetime_pre"]).abs().dt.total_seconds() / 60.0
        )

    def score(row):
        df_val = row.get("dF", 0)
        dm_val = row.get("dM", 0)
        return (
            2.0 * float(df_val != 1)
            + 2.0 * float(dm_val != 1)
            + 0.5 * float(df_val > 1)
            + 0.5 * float(dm_val > 1)
        )

    pairs["score"] = pairs.apply(score, axis=1)
    pairs = (
        pairs.sort_values(["scanner_site", "datetime_main", "score"])
        .groupby(["scanner_site", "path_main"], as_index=False)
        .first()
    )
    pairs = pairs.drop_duplicates(subset="fname_without_site_main")
    priority = ['41stVida1', '41stVida2', '41stVida3', '41Vida1', '41Vida2', '41Vida3']

    pairs_sorted = pairs.assign(
        sort_key=pairs['scanner_site'].apply(
            lambda x: priority.index(x) if x in priority else len(priority)
        )
    ).sort_values('sort_key').drop(columns='sort_key')


    paths = (
        pairs_sorted["path_main"].dropna().astype(str).apply(lambda p: p.lstrip("./"))
        if "path_main" in pairs_sorted
        else pd.Series(dtype=str)
    )
    path_list = paths.tolist()
    if max_files is not None and max_files > 0:
        path_list = path_list[:max_files]
    return path_list


def staging_size_bytes(staging: Path) -> int:
    """Return total bytes in staging, handling race conditions with workers."""
    total = 0

    def _onerror(err: OSError) -> None:
        if isinstance(err, FileNotFoundError):
            return
        raise err

    for root, dirs, files in os.walk(staging, onerror=_onerror):
        # Avoid transient lock directories while workers are running.
        dirs[:] = [d for d in dirs if not d.endswith(".lock")]
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).stat().st_size
            except (FileNotFoundError, OSError):
                # File was deleted by worker between listing and stat - skip it
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
    parser = argparse.ArgumentParser(description="Copy .dat files into staging for streaming recon")
    parser.add_argument("--config", default="config/streaming/dwi.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    transfer_cfg = cfg["transfer"]

    raw_max_files = transfer_cfg.get("max_files")
    max_files = int(raw_max_files) if raw_max_files not in (None, "null", "None") else None
    manifest_paths = build_main_manifest(
        Path(transfer_cfg["manifest_csv"]), transfer_cfg["min_bytes"], max_files=max_files
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
