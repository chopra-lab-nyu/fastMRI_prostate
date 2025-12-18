"""DWI ESC reconstruction from Siemens .dat files."""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import (
    dwi_reconstruction_diffusion,
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
DEFAULT_DIRECTIONS: Tuple[str, ...] = ALL_DIRECTIONS
DEFAULT_AVERAGING_SCHEMES: Tuple[Tuple[str, int, int], ...] = (
    ("b50_1_b1000_1", 1, 1),
    ("b50_1_b1000_2", 1, 2),
    ("b50_1_b1000_3", 1, 3),
    ("b50_2_b1000_6", 2, 6),
    ("b50_4_b1000_12", 4, 12),
)
DEFAULT_COMBINES: Tuple[str, ...] = ("rss", "espirit", "esc")
REQUIRED_FOR_METRICS = set(ALL_DIRECTIONS)
CENTER_CROP_SIZE = (100, 100)
VALID_AVERAGE_COUNTS = (48, 50)


class UnsupportedAverageCountError(RuntimeError):
    """Raised when a DWI scan has an unsupported number of averages."""


def parse_directions(value: str) -> List[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return list(DEFAULT_DIRECTIONS)

    invalid = [token for token in tokens if token not in ALL_DIRECTIONS]
    if invalid:
        raise ValueError(f"Unknown diffusion directions: {invalid}")

    return tokens


def parse_average_pair(value: str) -> Tuple[int, int]:
    tokens = [token.strip() for token in value.split(":") if token.strip()]
    if len(tokens) != 2:
        raise ValueError(f"Expected averages in 'low:high' format, got '{value}'")

    try:
        low_avg, high_avg = (int(tok) for tok in tokens)
    except ValueError as exc:  # noqa: PERF203
        raise ValueError(f"Non-integer averages in '{value}'") from exc

    if low_avg <= 0 or high_avg <= 0:
        raise ValueError(f"Averages must be positive, got {low_avg}:{high_avg}")
    if high_avg < low_avg:
        raise ValueError(f"High average must be >= low average, got {low_avg}:{high_avg}")

    return low_avg, high_avg


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


def _average_per_direction(
    per_average_vol: np.ndarray,
    direction_indices: Dict[str, np.ndarray],
    num_b50_averages: int,
    num_b1000_averages: int,
) -> Dict[str, np.ndarray]:
    """Average the per-average volume according to requested counts."""

    averaged_full = compute_averages(
        per_average_vol,
        num_b50_averages=num_b50_averages,
        num_b1000_averages=num_b1000_averages,
    )
    return {direction: averaged_full[direction] for direction in direction_indices}


def build_dwi_payload(
    recon_result,
    directions: Sequence[str],
    averaging_schemes: Sequence[Tuple[str, int, int]],
    compute_metrics: bool,
    combines: Sequence[str],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}

    # Shared metadata
    payload["metadata/directions"] = np.asarray(directions, dtype="S12")
    payload["metadata/averaging_schemes"] = np.asarray([tag for tag, _, _ in averaging_schemes], dtype="S20")
    available_combines: list[str] = []
    for name in combines:
        if name == "espirit" and recon_result.espirit_images_per_average is None:
            continue
        available_combines.append(name)
    payload["metadata/combines"] = np.asarray(available_combines, dtype="S16")

    for direction in directions:
        payload[f"metadata/direction_indices/{direction}"] = recon_result.direction_indices[direction].astype(np.int16)

    # Shared volumes
    payload["images/esc_full"] = recon_result.esc_images_per_average.astype(np.float32)
    payload["coil/post_grappa_full"] = np.abs(recon_result.post_grappa_coil_images).astype(np.float32)
    payload["kspace/post_grappa_full"] = recon_result.post_grappa_kspace.astype(np.complex64)

    combine_sources = {}
    if "esc" in available_combines:
        combine_sources["esc"] = recon_result.esc_images_per_average
    if "rss" in available_combines:
        combine_sources["rss"] = recon_result.rss_images_per_average
    if "espirit" in available_combines and recon_result.espirit_images_per_average is not None:
        combine_sources["espirit"] = recon_result.espirit_images_per_average

    for combine_name, per_average_vol in combine_sources.items():
        for direction in directions:
            payload[f"images/{combine_name}/per_average/{direction}"] = (
                per_average_vol[recon_result.direction_indices[direction]].astype(np.float32)
            )

        # ESC gets single-coil k-space per direction
        if combine_name == "esc":
            for direction in directions:
                payload[f"kspace/esc/{direction}"] = recon_result.get_esc_direction_kspace(direction).astype(np.complex64)

        for tag, b50_avg, b1000_avg in averaging_schemes:
            averaged = _average_per_direction(per_average_vol, recon_result.direction_indices, b50_avg, b1000_avg)
            for direction in directions:
                payload[f"images/{combine_name}/{tag}/{direction}"] = _postprocess_volume(averaged[direction])

            if compute_metrics and REQUIRED_FOR_METRICS.issubset(set(directions)):
                metrics_input = {key: averaged[key].copy() for key in REQUIRED_FOR_METRICS}
                metrics = compute_trace_adc_b1500(metrics_input)
                payload[f"metrics/{combine_name}/{tag}/adc_map"] = _postprocess_volume(metrics["adc_map"])
                payload[f"metrics/{combine_name}/{tag}/b1500"] = _postprocess_volume(metrics["b1500"])
                payload[f"metrics/{combine_name}/{tag}/trace_b50"] = _postprocess_volume(metrics["trace_b50"])
                payload[f"metrics/{combine_name}/{tag}/trace_b1000"] = _postprocess_volume(metrics["trace_b1000"])

    return payload


def process_dat_file(
    dat_file: Path,
    directions: Sequence[str],
    averaging_schemes: Sequence[Tuple[str, int, int]],
    output_dir: Path,
    skip_metrics: bool,
    combines: Sequence[str],
) -> Path:
    kspace, calibration, hdr = load_dat_file_dwi(dat_file)

    avg_count = kspace.shape[0]
    if avg_count not in VALID_AVERAGE_COUNTS:
        raise UnsupportedAverageCountError(
            f"{dat_file.name}: found {avg_count} averages; expected one of {VALID_AVERAGE_COUNTS}"
        )

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

    compute_metrics = not skip_metrics
    recon_result = dwi_reconstruction_diffusion(
        kspace,
        calibration,
        hdr,
        directions=directions,
        enable_espirit="espirit" in combines,
        compute_metrics=compute_metrics,
    )

    payload = build_dwi_payload(recon_result, directions, averaging_schemes, compute_metrics, combines)
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
    averaging_schemes = DEFAULT_AVERAGING_SCHEMES
    combines = [c.strip() for c in args.combines.split(",") if c.strip()]
    if not combines:
        combines = list(DEFAULT_COMBINES)

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
        try:
            output_path = process_dat_file(dat_file, directions, averaging_schemes, output_dir, args.skip_metrics, combines)
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
        "--combines",
        default=",".join(DEFAULT_COMBINES),
        help="Comma-separated coil combination methods to include (subset of esc,rss,espirit).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Limit the number of .dat files processed (for quick dev/test).",
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
