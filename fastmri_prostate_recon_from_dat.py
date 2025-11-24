"""DWI ESC reconstruction from Siemens .dat files."""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import (
    dwi_reconstruction_esc,
    compute_averages,
)
from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.utils import flip_im, center_crop_im
from fastmri_prostate.data.mri_data import load_dat_file_dwi, save_recon


ALL_DIRECTIONS: Tuple[str, ...] = (
    "b50x",
    "b50y",
    "b50z",
    "b1000x",
    "b1000y",
    "b1000z",
)
DEFAULT_DIRECTIONS: Tuple[str, ...] = ("b50x", "b1000x")
DEFAULT_AVERAGES: Tuple[int, int] = (4, 12)
REQUIRED_FOR_METRICS = set(ALL_DIRECTIONS)
CENTER_CROP_SIZE = (100, 100)


def parse_directions(value: str) -> List[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return list(DEFAULT_DIRECTIONS)

    invalid = [token for token in tokens if token not in ALL_DIRECTIONS]
    if invalid:
        raise ValueError(f"Unknown diffusion directions: {invalid}")

    return tokens


def parse_average_pair(value: str) -> Tuple[int, int]:
    try:
        b50_str, b1000_str = value.split(":", 1)
        return int(b50_str), int(b1000_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid average specification '{value}'. Expected '<b50>:<b1000>'."
        ) from exc


def _postprocess_volume(volume: np.ndarray) -> np.ndarray:
    processed = flip_im(volume.copy(), 0)
    processed = center_crop_im(processed, CENTER_CROP_SIZE)
    return processed.astype(np.float32)


def _sanitize_patient_id(value: Any) -> str:
    if value is None:
        return "unknown_patient"

    if isinstance(value, (int, np.integer)):
        cleaned = str(int(value))
    elif isinstance(value, (float, np.floating)):
        cleaned = str(int(value)) if float(value).is_integer() else f"{value}"
    else:
        cleaned = str(value).strip()

    if not cleaned:
        return "unknown_patient"
    return cleaned


def build_dwi_payload(
    esc_result,
    directions: Sequence[str],
    averages: Tuple[int, int],
    include_metrics: bool,
) -> dict[str, np.ndarray]:
    b50_averages, b1000_averages = averages
    averaged_full = compute_averages(
        esc_result.esc_images_per_average,
        num_b50_averages=b50_averages,
        num_b1000_averages=b1000_averages,
    )
    averaged = {direction: averaged_full[direction] for direction in directions}

    payload: dict[str, np.ndarray] = {}

    for direction in directions:
        payload[f"images/averaged/{direction}"] = _postprocess_volume(averaged[direction])
        payload[f"images/per_average/{direction}"] = (
            esc_result.esc_images_per_average[esc_result.direction_indices[direction]].astype(np.float32)
        )
        payload[f"kspace/esc/{direction}"] = esc_result.get_direction_kspace(direction).astype(np.complex64)
        payload[f"metadata/direction_indices/{direction}"] = esc_result.direction_indices[direction].astype(np.int16)

    if include_metrics:
        metrics_input = {key: averaged[key].copy() for key in REQUIRED_FOR_METRICS}
        metrics = compute_trace_adc_b1500(metrics_input)
        payload["metrics/adc_map"] = _postprocess_volume(metrics["adc_map"])
        payload["metrics/b1500"] = _postprocess_volume(metrics["b1500"])
        payload["metrics/trace_b50"] = _postprocess_volume(metrics["trace_b50"])
        payload["metrics/trace_b1000"] = _postprocess_volume(metrics["trace_b1000"])

    payload["images/esc_full"] = esc_result.esc_images_per_average.astype(np.float32)
    payload["kspace/post_grappa_full"] = esc_result.kspace_esc.astype(np.complex64)
    payload["metadata/directions"] = np.asarray(directions, dtype="S12")
    payload["metadata/averages"] = np.asarray(averages, dtype=np.int16)
    return payload


def process_dat_file(
    dat_file: Path,
    directions: Sequence[str],
    averages: Tuple[int, int],
    output_dir: Path,
    skip_metrics: bool,
) -> Path:
    kspace, calibration, hdr = load_dat_file_dwi(dat_file)

    patient_id_raw = None
    if isinstance(hdr, dict):
        patient_id_raw = hdr.pop("patient_id", None)
    if patient_id_raw is None:
        logging.warning("Missing PatientID in header for %s", dat_file.name)
        patient_id = "unknown_patient"
    else:
        if isinstance(patient_id_raw, bytes):
            patient_id_raw = patient_id_raw.decode(errors="ignore")
        patient_id = _sanitize_patient_id(patient_id_raw)

    compute_metrics = not skip_metrics and REQUIRED_FOR_METRICS.issubset(set(directions))
    esc_result = dwi_reconstruction_esc(
        kspace,
        calibration,
        hdr,
        num_b50_averages=averages[0],
        num_b1000_averages=averages[1],
        directions=directions,
        compute_metrics=compute_metrics,
    )

    payload = build_dwi_payload(esc_result, directions, averages, compute_metrics)
    base_stem = dat_file.stem
    output_name = f"{base_stem}__{patient_id}.h5"
    output_path = output_dir / output_name
    save_recon(payload, hdr, output_path)
    return output_path


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    directions = parse_directions(args.directions)
    averages = parse_average_pair(args.averages)

    dat_files = sorted(data_dir.glob("*.dat"))
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
        try:
            output_path = process_dat_file(dat_file, directions, averages, output_dir, args.skip_metrics)
        except Exception:  # noqa: BLE001
            logging.exception("Failed to reconstruct %s", dat_file.name)
            continue
        logging.info("Saved %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESC DWI reconstruction on Siemens .dat files")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing Siemens DWI .dat files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reconstructed HDF5 files will be written",
    )
    parser.add_argument(
        "--directions",
        default=",".join(DEFAULT_DIRECTIONS),
        help=(
            "Comma-separated diffusion directions to keep "
            f"(default: {','.join(DEFAULT_DIRECTIONS)})"
        ),
    )
    parser.add_argument(
        "--averages",
        default=f"{DEFAULT_AVERAGES[0]}:{DEFAULT_AVERAGES[1]}",
        help="Average specification as <b50>:<b1000> (default: 4:12)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip computing trace/ADC/b1500 maps",
    )
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


if __name__ == "__main__":
    main()