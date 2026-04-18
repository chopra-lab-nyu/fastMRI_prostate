"""T2 GRAPPA reconstruction from Siemens .dat files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from fastmri_prostate.data.mri_data import load_dat_file_T2, save_recon
from fastmri_prostate.reconstruction.t2.prostate_t2_recon import (
    DEFAULT_T2_AVERAGING_SCHEMES,
    t2_reconstruction,
)

DEFAULT_AVERAGING_SCHEMES: Tuple[Tuple[str, Tuple[int, ...]], ...] = DEFAULT_T2_AVERAGING_SCHEMES
VALID_AVERAGES = (1, 2, 3)


class UnsupportedAverageCountError(RuntimeError):
    """Raised when a T2 scan has incompatible average count for requested schemes."""


def _canonicalize_scheme(values: Sequence[int]) -> Tuple[int, ...]:
    scheme = tuple(sorted({int(v) for v in values}))
    if not scheme:
        raise ValueError("Average scheme cannot be empty.")
    if any(v not in VALID_AVERAGES for v in scheme):
        raise ValueError(f"Average indices must be within {VALID_AVERAGES}, got {scheme}")
    return scheme


def _scheme_tag(scheme: Sequence[int]) -> str:
    return f"axt2_{'_'.join(str(v) for v in scheme)}"


def parse_average_schemes(raw_value: Any) -> List[Tuple[str, Tuple[int, ...]]]:
    if raw_value in (None, [], "all", "ALL"):
        return list(DEFAULT_AVERAGING_SCHEMES)

    schemes: List[Tuple[str, Tuple[int, ...]]] = []
    seen = set()

    if isinstance(raw_value, str):
        tokens = [tok.strip() for tok in raw_value.split(";") if tok.strip()]
        if not tokens:
            return list(DEFAULT_AVERAGING_SCHEMES)
        parsed_schemes = []
        for token in tokens:
            parsed_schemes.append(_canonicalize_scheme([int(v) for v in token.split(",") if v.strip()]))
    elif isinstance(raw_value, (list, tuple)):
        if raw_value and isinstance(raw_value[0], (list, tuple)):
            parsed_schemes = [_canonicalize_scheme(item) for item in raw_value]
        else:
            parsed_schemes = [_canonicalize_scheme(raw_value)]
    else:
        raise ValueError(f"Unsupported averages format: {type(raw_value)}")

    for scheme in parsed_schemes:
        if scheme in seen:
            continue
        schemes.append((_scheme_tag(scheme), scheme))
        seen.add(scheme)

    if not schemes:
        raise ValueError("No valid averaging schemes provided.")

    return schemes


def _sanitize_patient_id(value: Any) -> str:
    if value is None:
        return "unknown_patient"

    if isinstance(value, (int, np.integer)):
        cleaned = str(int(value))
    elif isinstance(value, (float, np.floating)):
        cleaned = str(int(value)) if float(value).is_integer() else f"{value}"
    else:
        cleaned = str(value).strip()

    return cleaned if cleaned else "unknown_patient"


def process_dat_file(
    dat_file: Path,
    averaging_schemes: Sequence[Tuple[str, Sequence[int]]],
    output_dir: Path,
    single_average_output_dir: Path | None = None,
    store_kspace: bool = True,
) -> Path:
    kspace, calibration, hdr = load_dat_file_T2(dat_file)

    avg_count = int(kspace.shape[0])
    valid_schemes = [(tag, tuple(indices)) for tag, indices in averaging_schemes if max(indices) <= avg_count]
    if not valid_schemes:
        raise UnsupportedAverageCountError(
            f"{dat_file.name}: found {avg_count} averages; no averaging scheme is valid for this file"
        )

    patient_id = None
    if isinstance(hdr, dict):
        patient_id = hdr.get("Config", {}).get("PatientID")
    if isinstance(patient_id, bytes):
        patient_id = patient_id.decode(errors="ignore")
    patient_id = _sanitize_patient_id(patient_id)

    payload = t2_reconstruction(
        kspace,
        calibration,
        hdr,
        averaging_schemes=valid_schemes,
        store_kspace=store_kspace,
    )

    selected_output_dir = single_average_output_dir if avg_count == 1 and single_average_output_dir is not None else output_dir
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{dat_file.stem}__{patient_id}.h5"
    output_path = selected_output_dir / output_name
    save_recon(payload, hdr, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T2 reconstruction on Siemens .dat files")
    parser.add_argument("--data-dir", required=True, help="Directory containing T2 .dat files")
    parser.add_argument("--output-dir", required=True, help="Output directory for reconstructed .h5 files")
    parser.add_argument(
        "--averages",
        default="all",
        help="Averaging schemes as 'all' or semicolon list like '1;2;1,2;2,3;1,3;1,2,3'",
    )
    parser.add_argument(
        "--skip-kspace",
        action="store_true",
        help="Skip writing k-space datasets in output HDF5",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files to process")
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Zero-based index for this job (defaults to 0 if unset)",
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=1,
        help="Total number of jobs used to split the workload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    averaging_schemes = parse_average_schemes(args.averages)

    dat_files = sorted(data_dir.glob("*.dat"))
    if args.max_files is not None and args.max_files > 0:
        dat_files = dat_files[: args.max_files]
    if not dat_files:
        logging.warning("No .dat files found in %s", data_dir)
        return

    job_count = args.job_count if args.job_count and args.job_count > 0 else 1
    job_index = args.job_index if args.job_index is not None else 0

    total_files = len(dat_files)
    files_per_job = (total_files + job_count - 1) // job_count
    start = job_index * files_per_job
    end = min(start + files_per_job, total_files)

    if start >= total_files or start >= end:
        logging.info(
            "Job %d has no assigned files (job count %d, total files %d)",
            job_index,
            job_count,
            total_files,
        )
        return

    assigned_files = dat_files[start:end]
    logging.info(
        "Job %d/%d handling files %d-%d (total %d)",
        job_index,
        job_count,
        start,
        end - 1,
        total_files,
    )

    for dat_file in assigned_files:
        logging.info("Processing %s", dat_file.name)
        output_path = process_dat_file(
            dat_file=dat_file,
            averaging_schemes=averaging_schemes,
            output_dir=output_dir,
            store_kspace=not args.skip_kspace,
        )
        logging.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
