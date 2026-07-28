"""Build compact ESPIRiT DWI state-bank H5 files directly from Siemens DAT files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import yaml

from fastmri_prostate.data.mri_data import load_dat_file_dwi
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import dwi_reconstruction_diffusion
from scripts.streaming.dwi.build_espirit_state_bank_h5 import (
    ALL_DIRECTIONS,
    COMBINE_NAME,
    EXPECTED_DIRECTION_COUNTS,
    _build_cumulative_mean,
    _build_metadata,
    _validate_file_structure,
    _write_dataset,
)
from scripts.streaming.dwi.recon_from_dat import (
    UnsupportedAverageCountError,
    VALID_AVERAGE_COUNTS,
    _sanitize_patient_id,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

DEFAULT_CONFIG = Path("config/streaming/dwi_direct_state_bank.yaml")
STANDARD_MODALITY = "AXDIFFUSION_HVD"


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text())
    return loaded or {}


def _config_path(cfg: dict[str, Any], section: str, key: str) -> Path | None:
    value = cfg.get(section, {}).get(key)
    return Path(value) if value not in (None, "", "None", "null") else None


def _bank_source_stems(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    stems: set[str] = set()
    for h5_path in output_dir.glob("*.h5"):
        stem = h5_path.stem
        stems.add(stem.rsplit("__", 1)[0] if "__" in stem else stem)
    return stems


def _read_manifest_paths(manifest_csv: Path, source_root: Path | None) -> list[Path]:
    dat_paths: list[Path] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError(f"{manifest_csv} must contain a 'path' column.")
        for row in reader:
            path_value = row["path"].strip()
            if not path_value:
                continue
            path = Path(path_value)
            if not path.is_absolute():
                if source_root is None:
                    raise ValueError("Relative manifest paths require --source-root or manifest.source_root in config.")
                path = source_root / path_value.lstrip("./")
            dat_paths.append(path)
    return dat_paths


def _select_assigned_files(dat_files: Sequence[Path], job_index: int, job_count: int) -> tuple[list[Path], int, int]:
    total_files = len(dat_files)
    files_per_job = (total_files + job_count - 1) // job_count
    start = job_index * files_per_job
    end = min(start + files_per_job, total_files)
    return list(dat_files[start:end]), start, end


def _extract_patient_id(hdr: dict[str, Any], dat_file: Path) -> str:
    patient_id_raw = hdr.pop("patient_id", None)
    if patient_id_raw is None:
        logging.warning("Missing PatientID in header for %s", dat_file.name)
        return "unknown_patient"
    if isinstance(patient_id_raw, bytes):
        patient_id_raw = patient_id_raw.decode(errors="ignore")
    return _sanitize_patient_id(patient_id_raw)


def _is_standard_dat(dat_file: Path) -> bool:
    return dat_file.stem.split("#")[-1] == STANDARD_MODALITY


def build_state_bank_h5_from_dat(
    dat_file: Path,
    output_dir: Path,
    legacy_source_h5_dir: Path | None,
    overwrite: bool,
    enable_phasecorr: bool,
) -> tuple[Path, int]:
    if not _is_standard_dat(dat_file):
        raise ValueError(f"Nonstandard DWI DAT is out of scope: {dat_file.name}")

    phasecorr = None
    if enable_phasecorr:
        kspace, calibration, hdr, phasecorr = load_dat_file_dwi(dat_file, include_phasecorr=True)
    else:
        kspace, calibration, hdr = load_dat_file_dwi(dat_file)

    avg_count = kspace.shape[0]
    if avg_count not in VALID_AVERAGE_COUNTS:
        raise UnsupportedAverageCountError(
            f"{dat_file.name}: found {avg_count} averages; expected one of {VALID_AVERAGE_COUNTS}"
        )

    patient_id = _extract_patient_id(hdr, dat_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / f"{dat_file.stem}__{patient_id}.h5"
    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_h5_path}")

    recon = dwi_reconstruction_diffusion(
        kspace,
        calibration,
        hdr,
        phasecorr=phasecorr,
        directions=ALL_DIRECTIONS,
        compute_metrics=False,
        enable_esc=False,
        enable_espirit=True,
        enable_phasecorr=enable_phasecorr,
        enable_rss=False,
        store_post_grappa=False,
    )
    if recon.espirit_images_per_average is None:
        raise RuntimeError(f"ESPIRiT images were not produced for {dat_file.name}.")

    metadata_source = (legacy_source_h5_dir / output_h5_path.name) if legacy_source_h5_dir is not None else output_h5_path
    payload = _build_metadata(metadata_source, json.dumps(hdr))
    num_slices: int | None = None
    for direction in ALL_DIRECTIONS:
        expected_count = EXPECTED_DIRECTION_COUNTS[direction]
        indices = recon.direction_indices[direction]
        if len(indices) != expected_count:
            raise ValueError(f"Expected {expected_count} averages for {direction}, found {len(indices)}.")
        cumulative_mean = _build_cumulative_mean(recon.espirit_images_per_average[indices])
        shell = "b50" if direction.startswith("b50") else "b1000"
        axis = direction[-1]
        payload[f"bank/{COMBINE_NAME}/{shell}/{axis}"] = cumulative_mean
        if num_slices is None:
            num_slices = int(cumulative_mean.shape[1])

    temp_path = output_h5_path.with_name(f".{output_h5_path.name}.direct_state_bank_tmp_{os.getpid()}")
    try:
        with h5py.File(temp_path, "w") as output_hf:
            for key, value in sorted(payload.items()):
                _write_dataset(output_hf, key, value)
        _validate_file_structure(temp_path)
        if output_h5_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_h5_path}")
        temp_path.replace(output_h5_path)
        return output_h5_path, int(num_slices or 0)
    finally:
        temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dat-file", type=Path, default=None, help="Process a single DAT file instead of a manifest.")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-h5-dir-for-metadata", type=Path, default=None)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-phasecorr", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_config(args.config)

    manifest_csv = args.manifest_csv or _config_path(cfg, "manifest", "csv")
    source_root = args.source_root or _config_path(cfg, "manifest", "source_root")
    output_dir = args.output_dir or _config_path(cfg, "process", "output_dir")
    legacy_source_h5_dir = args.source_h5_dir_for_metadata or _config_path(cfg, "process", "source_h5_dir_for_metadata")
    overwrite = bool(cfg.get("process", {}).get("overwrite", False)) or args.overwrite
    enable_phasecorr = bool(cfg.get("process", {}).get("enable_phasecorr", False)) or args.enable_phasecorr

    if output_dir is None:
        raise ValueError("Provide --output-dir or process.output_dir in config.")

    if args.dat_file is not None:
        dat_files = [args.dat_file]
    else:
        if manifest_csv is None:
            raise ValueError("Provide --manifest-csv, --dat-file, or manifest.csv in config.")
        dat_files = _read_manifest_paths(manifest_csv, source_root)

    if args.limit is not None and args.limit > 0:
        dat_files = dat_files[: args.limit]

    assigned_files, start, end = _select_assigned_files(dat_files, args.job_index, max(1, args.job_count))
    logging.info(
        "Processing %d DAT files from indices [%d, %d) out of %d total files",
        len(assigned_files),
        start,
        end,
        len(dat_files),
    )

    existing_stems = _bank_source_stems(output_dir)
    ok = skipped = errors = 0
    for dat_file in assigned_files:
        if dat_file.stem in existing_stems and not overwrite:
            logging.info("Skipping already banked DAT stem %s", dat_file.stem)
            skipped += 1
            continue
        if not dat_file.exists():
            logging.warning("Skipping missing DAT %s", dat_file)
            skipped += 1
            continue

        try:
            output_path, num_slices = build_state_bank_h5_from_dat(
                dat_file,
                output_dir,
                legacy_source_h5_dir,
                overwrite=overwrite,
                enable_phasecorr=enable_phasecorr,
            )
            existing_stems.add(dat_file.stem)
            logging.info("Built %s slices=%d", output_path, num_slices)
            ok += 1
        except (FileExistsError, UnsupportedAverageCountError, ValueError) as exc:
            logging.warning("Skipping %s: %s", dat_file.name, exc)
            skipped += 1
        except Exception:
            logging.exception("Failed %s", dat_file)
            errors += 1

    logging.info("Finished shard %d/%d: %d ok, %d skipped, %d errors", args.job_index, args.job_count, ok, skipped, errors)


if __name__ == "__main__":
    main()
