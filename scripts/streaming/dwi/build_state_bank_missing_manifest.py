"""Build a DAT manifest for DWI cases missing ESPIRiT state-bank H5s."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from scripts.streaming.dwi.transfer import MAIN_MIN_BYTES, human_to_bytes


DEFAULT_INPUT_CSV = Path("/gpfs/data/prostatelab/processed_data/csv/kspace_prostate_dwi_file_metadata_yarra_axdiffusion_hvd.csv")
DEFAULT_OUTPUT_CSV = Path("/gpfs/data/prostatelab/processed_data/csv/kspace_prostate_dwi_state_bank_missing_axdiffusion_hvd.csv")
DEFAULT_BANK_DIR = Path("/gpfs/scratch/td2105/dwi_stream/recons_espirit_state_bank_v2")
STANDARD_MODALITY = "AXDIFFUSION_HVD"


def _dat_stem(path_value: str) -> str:
    return Path(path_value).stem


def _bank_source_stem(h5_path: Path) -> str:
    stem = h5_path.stem
    return stem.rsplit("__", 1)[0] if "__" in stem else stem


def _is_standard_dat_path(path_value: str) -> bool:
    return Path(path_value).stem.split("#")[-1] == STANDARD_MODALITY


def _read_manifest(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a 'path' column.")
        return [dict(row) for row in reader]


def _write_manifest(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["size", "path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_missing_rows(rows: list[dict[str, str]], bank_dir: Path, min_bytes: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    existing_stems = {_bank_source_stem(path) for path in bank_dir.glob("*.h5")}
    seen_dat_stems: set[str] = set()
    missing_rows: list[dict[str, str]] = []
    counts = {
        "total": 0,
        "standard": 0,
        "too_small": 0,
        "nonstandard": 0,
        "duplicates": 0,
        "banked": 0,
        "missing": 0,
    }

    for row in rows:
        counts["total"] += 1
        path_value = row.get("path", "")
        if not _is_standard_dat_path(path_value):
            counts["nonstandard"] += 1
            continue
        counts["standard"] += 1
        if not human_to_bytes(row.get("size", "")) >= min_bytes:
            counts["too_small"] += 1
            continue

        stem = _dat_stem(path_value)
        if stem in seen_dat_stems:
            counts["duplicates"] += 1
            continue
        seen_dat_stems.add(stem)

        if stem in existing_stems:
            counts["banked"] += 1
        else:
            missing_rows.append(row)
            counts["missing"] += 1

    return missing_rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--min-bytes", type=int, default=int(MAIN_MIN_BYTES))
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing the missing manifest.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    rows = _read_manifest(args.manifest_csv)
    missing_rows, counts = build_missing_rows(rows, args.bank_dir, args.min_bytes)
    logging.info(
        (
            "Manifest total=%d standard=%d too_small=%d banked=%d missing=%d "
            "duplicates=%d nonstandard=%d min_bytes=%d bank_dir=%s"
        ),
        counts["total"],
        counts["standard"],
        counts["too_small"],
        counts["banked"],
        counts["missing"],
        counts["duplicates"],
        counts["nonstandard"],
        args.min_bytes,
        args.bank_dir,
    )

    if args.dry_run:
        return

    _write_manifest(args.output_csv, missing_rows)
    logging.info("Wrote %d missing rows to %s", len(missing_rows), args.output_csv)


if __name__ == "__main__":
    main()
