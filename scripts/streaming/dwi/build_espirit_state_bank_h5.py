"""Build compact ESPIRiT-only DWI state-bank H5 files from existing recon H5s."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import h5py
import numpy as np
import yaml

from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import get_direction_indices
from fastmri_prostate.reconstruction.utils import center_crop_im, flip_im
from scripts.streaming.dwi.recon_from_dat import DEFAULT_AVERAGING_SCHEMES


os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


COMBINE_NAME = "espirit"
BANK_VERSION = "dwi_espirit_state_bank_v2"
CENTER_CROP_SIZE = (100, 100)
ADC_SCALE = -1e6
B_VALUES = (50, 1000)
ALL_DIRECTIONS = ("b50x", "b50y", "b50z", "b1000x", "b1000y", "b1000z")
EXPECTED_DIRECTION_COUNTS = {
    "b50x": 4,
    "b50y": 4,
    "b50z": 4,
    "b1000x": 12,
    "b1000y": 12,
    "b1000z": 12,
}
LEGACY_TAGS: tuple[str, ...] = tuple(tag for tag, _, _ in DEFAULT_AVERAGING_SCHEMES)
LEGACY_TAG_TO_COUNTS: dict[str, tuple[int, int]] = {tag: (b50_count, b1000_count) for tag, b50_count, b1000_count in DEFAULT_AVERAGING_SCHEMES}
VALIDATION_TAGS: tuple[str, ...] = ("b50_1_b1000_1", "b50_2_b1000_2", "b50_4_b1000_12", "b50_2_b1000_6")
IMAGE_VALIDATION_ATOL = 1e-5
METRIC_VALIDATION_ATOL = {
    "trace_b50": 1e-5,
    "trace_b1000": 1e-5,
    "adc_map": 2e-4,
    "b1500": 1e-6,
}


def _decode_string_values(dataset_value: np.ndarray) -> list[str]:
    if isinstance(dataset_value, bytes):
        return [dataset_value.decode(errors="ignore")]
    if isinstance(dataset_value, np.ndarray) and dataset_value.dtype.kind == "S":
        return [item.decode(errors="ignore") for item in dataset_value.tolist()]
    return [str(item) for item in np.atleast_1d(dataset_value).tolist()]


def _load_process_config(
    config_path: Path,
) -> tuple[Path | None, Path | None, str | None, bool | None, Path | None]:
    cfg = yaml.safe_load(config_path.read_text())
    process_cfg = cfg["process"]
    input_dir = Path(process_cfg["input_dir"]) if process_cfg.get("input_dir") is not None else None
    output_dir = Path(process_cfg["output_dir"]) if process_cfg.get("output_dir") is not None else None
    glob_pattern = str(process_cfg["glob"]) if process_cfg.get("glob") is not None else None
    overwrite = bool(process_cfg["overwrite"]) if process_cfg.get("overwrite") is not None else None
    manifest_path = Path(process_cfg["manifest_path"]) if process_cfg.get("manifest_path") is not None else None
    return input_dir, output_dir, glob_pattern, overwrite, manifest_path


def _resolve_runtime_args(args: argparse.Namespace) -> tuple[Path, Path, str, bool, Path | None]:
    config_input_dir: Path | None = None
    config_output_dir: Path | None = None
    config_glob: str | None = None
    config_overwrite: bool | None = None
    config_manifest_path: Path | None = None
    if args.config is not None:
        (
            config_input_dir,
            config_output_dir,
            config_glob,
            config_overwrite,
            config_manifest_path,
        ) = _load_process_config(args.config)

    input_dir = Path(args.input_dir) if args.input_dir is not None else config_input_dir
    output_dir = Path(args.output_dir) if args.output_dir is not None else config_output_dir
    glob_pattern = args.glob if args.glob is not None else config_glob
    overwrite = bool(args.overwrite) if args.overwrite else bool(config_overwrite)
    manifest_path = Path(args.manifest_path) if args.manifest_path is not None else config_manifest_path

    if input_dir is None:
        raise ValueError("Provide --input-dir or --config.")
    if output_dir is None:
        raise ValueError("Provide --output-dir or --config.")
    if glob_pattern is None:
        glob_pattern = "*.h5"

    return input_dir, output_dir, glob_pattern, overwrite, manifest_path


def _list_h5_files(input_dir: Path, glob_pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.glob(glob_pattern) if path.is_file())


def _select_assigned_files(h5_files: Sequence[Path], job_index: int, job_count: int) -> tuple[list[Path], int, int]:
    total_files = len(h5_files)
    files_per_job = (total_files + job_count - 1) // job_count
    start = job_index * files_per_job
    end = min(start + files_per_job, total_files)
    return list(h5_files[start:end]), start, end


def _postprocess_average(volume: np.ndarray) -> np.ndarray:
    processed = flip_im(volume.copy(), 0)
    processed = center_crop_im(processed, CENTER_CROP_SIZE)
    return processed.astype(np.float32, copy=False)


def _build_cumulative_mean(per_average_vol: np.ndarray) -> np.ndarray:
    per_average_vol = per_average_vol.astype(np.float32, copy=False)
    cumulative_sum = np.cumsum(per_average_vol, axis=0, dtype=np.float32)
    divisors = np.arange(1, per_average_vol.shape[0] + 1, dtype=np.float32).reshape(-1, 1, 1, 1)
    return (cumulative_sum / divisors).astype(np.float32, copy=False)


def _get_fixed_acquisition_order() -> tuple[np.ndarray, np.ndarray]:
    direction_indices = get_direction_indices(48)
    reverse_map: dict[int, tuple[str, int]] = {}
    for direction, indices in direction_indices.items():
        for cumulative_count, index in enumerate(indices, start=1):
            reverse_map[int(index)] = (direction, cumulative_count)

    ordered = [reverse_map[idx] for idx in sorted(reverse_map)]
    direction_values = np.asarray([direction for direction, _ in ordered], dtype="S12")
    cumulative_counts = np.asarray([count for _, count in ordered], dtype=np.int16)
    return direction_values, cumulative_counts


def _build_metadata(source_h5_path: Path, hdr_dataset: Any) -> dict[str, Any]:
    direction_values, cumulative_counts = _get_fixed_acquisition_order()
    return {
        "metadata/version": BANK_VERSION,
        "metadata/combine": COMBINE_NAME,
        "metadata/directions": np.asarray(ALL_DIRECTIONS, dtype="S12"),
        "metadata/max_counts": np.asarray([EXPECTED_DIRECTION_COUNTS[direction] for direction in ALL_DIRECTIONS], dtype=np.int16),
        "metadata/b_values": np.asarray(B_VALUES, dtype=np.int16),
        "metadata/adc_scale": np.asarray(ADC_SCALE, dtype=np.float32),
        "metadata/crop_size": np.asarray(CENTER_CROP_SIZE, dtype=np.int16),
        "metadata/source_h5_basename": source_h5_path.name,
        "metadata/source_h5_path": str(source_h5_path),
        "metadata/legacy_balanced_tags": np.asarray(LEGACY_TAGS, dtype="S20"),
        "metadata/acquisition_order/direction": direction_values,
        "metadata/acquisition_order/cumulative_count": cumulative_counts,
        "hdr": hdr_dataset,
    }


def _ensure_parent_group(hf: h5py.File, dataset_path: str) -> None:
    current: h5py.Group | h5py.File = hf
    for part in dataset_path.split("/")[:-1]:
        current = current.require_group(part)


def _write_dataset(hf: h5py.File, dataset_path: str, value: Any) -> None:
    _ensure_parent_group(hf, dataset_path)
    if isinstance(value, np.ndarray) and value.ndim == 4 and dataset_path.startswith("bank/"):
        chunk_shape = (1, value.shape[1], value.shape[2], value.shape[3])
        hf.create_dataset(dataset_path, data=value, chunks=chunk_shape)
    else:
        hf.create_dataset(dataset_path, data=value)


def _load_espirit_per_average(source_hf: h5py.File) -> dict[str, np.ndarray]:
    available_combines = set(_decode_string_values(source_hf["metadata/combines"][()])) if "metadata/combines" in source_hf else set()
    if COMBINE_NAME not in available_combines:
        raise ValueError(f"Combine '{COMBINE_NAME}' not listed in metadata/combines.")

    per_average_by_direction: dict[str, np.ndarray] = {}
    for direction in ALL_DIRECTIONS:
        dataset_path = f"images/{COMBINE_NAME}/per_average/{direction}"
        if dataset_path not in source_hf:
            raise KeyError(f"Missing required dataset '{dataset_path}'.")
        per_average_vol = source_hf[dataset_path][()]
        expected_count = EXPECTED_DIRECTION_COUNTS[direction]
        if per_average_vol.shape[0] != expected_count:
            raise ValueError(
                f"Expected {expected_count} averages for {direction}, found {per_average_vol.shape[0]}."
            )
        per_average_by_direction[direction] = per_average_vol
    return per_average_by_direction


def _build_bank_payload(source_h5_path: Path) -> tuple[dict[str, Any], int]:
    with h5py.File(source_h5_path, "r") as source_hf:
        per_average_by_direction = _load_espirit_per_average(source_hf)
        hdr_dataset = source_hf["hdr"][()] if "hdr" in source_hf else json.dumps({})
        payload = _build_metadata(source_h5_path, hdr_dataset)
        num_slices: int | None = None

        for direction, per_average_vol in per_average_by_direction.items():
            shell = "b50" if direction.startswith("b50") else "b1000"
            axis = direction[-1]
            cumulative_mean = _build_cumulative_mean(per_average_vol)
            payload[f"bank/{COMBINE_NAME}/{shell}/{axis}"] = cumulative_mean
            if num_slices is None:
                num_slices = int(cumulative_mean.shape[1])

    if num_slices is None:
        raise RuntimeError(f"No bank datasets were built from {source_h5_path.name}.")
    return payload, num_slices


def _validate_file_structure(output_h5_path: Path) -> None:
    expected_bank_shapes = {
        "bank/espirit/b50/x": 4,
        "bank/espirit/b50/y": 4,
        "bank/espirit/b50/z": 4,
        "bank/espirit/b1000/x": 12,
        "bank/espirit/b1000/y": 12,
        "bank/espirit/b1000/z": 12,
    }

    with h5py.File(output_h5_path, "r") as bank_hf:
        for metadata_key in (
            "metadata/version",
            "metadata/combine",
            "metadata/directions",
            "metadata/max_counts",
            "metadata/b_values",
            "metadata/adc_scale",
            "metadata/crop_size",
            "metadata/source_h5_basename",
            "metadata/source_h5_path",
            "metadata/legacy_balanced_tags",
            "metadata/acquisition_order/direction",
            "metadata/acquisition_order/cumulative_count",
            "hdr",
        ):
            if metadata_key not in bank_hf:
                raise RuntimeError(f"Missing required dataset '{metadata_key}' in {output_h5_path.name}.")

        if _decode_string_values(bank_hf["metadata/directions"][()]) != list(ALL_DIRECTIONS):
            raise RuntimeError(f"Unexpected direction list in {output_h5_path.name}.")
        if _decode_string_values(bank_hf["metadata/legacy_balanced_tags"][()]) != list(LEGACY_TAGS):
            raise RuntimeError(f"Unexpected legacy tag list in {output_h5_path.name}.")

        for dataset_path, expected_len in expected_bank_shapes.items():
            if dataset_path not in bank_hf:
                raise RuntimeError(f"Missing bank dataset '{dataset_path}' in {output_h5_path.name}.")
            dataset = bank_hf[dataset_path]
            if dataset.shape[0] != expected_len:
                raise RuntimeError(
                    f"Unexpected first dimension for {dataset_path}: expected {expected_len}, found {dataset.shape[0]}."
                )
            if dataset.ndim != 4:
                raise RuntimeError(f"Expected 4D bank dataset for {dataset_path}, found shape {dataset.shape}.")


def _reconstruct_direction_images(bank_hf: h5py.File, tag: str) -> dict[str, np.ndarray]:
    try:
        b50_count, b1000_count = LEGACY_TAG_TO_COUNTS[tag]
    except KeyError as exc:
        raise ValueError(f"Unknown legacy tag '{tag}'.") from exc

    return {
        "b50x": bank_hf[f"bank/{COMBINE_NAME}/b50/x"][b50_count - 1],
        "b50y": bank_hf[f"bank/{COMBINE_NAME}/b50/y"][b50_count - 1],
        "b50z": bank_hf[f"bank/{COMBINE_NAME}/b50/z"][b50_count - 1],
        "b1000x": bank_hf[f"bank/{COMBINE_NAME}/b1000/x"][b1000_count - 1],
        "b1000y": bank_hf[f"bank/{COMBINE_NAME}/b1000/y"][b1000_count - 1],
        "b1000z": bank_hf[f"bank/{COMBINE_NAME}/b1000/z"][b1000_count - 1],
    }


def _validate_against_legacy_outputs(output_h5_path: Path, source_h5_path: Path) -> None:
    with h5py.File(output_h5_path, "r") as bank_hf, h5py.File(source_h5_path, "r") as source_hf:
        for tag in VALIDATION_TAGS:
            reconstructed_full = _reconstruct_direction_images(bank_hf, tag)
            reconstructed_postprocessed = {
                direction: _postprocess_average(image)
                for direction, image in reconstructed_full.items()
            }
            for direction, reconstructed_img in reconstructed_postprocessed.items():
                source_path = f"images/{COMBINE_NAME}/{tag}/{direction}"
                if source_path not in source_hf:
                    raise RuntimeError(f"Missing legacy validation dataset '{source_path}' in {source_h5_path.name}.")
                source_img = source_hf[source_path][()]
                if not np.allclose(reconstructed_img, source_img, atol=IMAGE_VALIDATION_ATOL, rtol=0):
                    raise RuntimeError(f"Directional image mismatch for tag '{tag}', direction '{direction}'.")

            metrics = {
                metric_name: _postprocess_average(metric_value)
                for metric_name, metric_value in compute_trace_adc_b1500(
                    {direction: image.copy() for direction, image in reconstructed_full.items()}
                ).items()
                if metric_name in ("trace_b50", "trace_b1000", "adc_map", "b1500")
            }
            for metric_name in ("trace_b50", "trace_b1000", "adc_map", "b1500"):
                source_metric_path = f"metrics/{COMBINE_NAME}/{tag}/{metric_name}"
                if source_metric_path not in source_hf:
                    raise RuntimeError(f"Missing legacy metric dataset '{source_metric_path}' in {source_h5_path.name}.")
                source_metric = source_hf[source_metric_path][()]
                if not np.allclose(metrics[metric_name], source_metric, atol=METRIC_VALIDATION_ATOL[metric_name], rtol=0):
                    raise RuntimeError(f"Metric mismatch for tag '{tag}', metric '{metric_name}'.")


def _validate_output(output_h5_path: Path, source_h5_path: Path) -> None:
    _validate_file_structure(output_h5_path)
    _validate_against_legacy_outputs(output_h5_path, source_h5_path)


def _write_manifest_row(manifest_path: Path | None, row: dict[str, Any]) -> None:
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_state_bank_h5(source_h5_path: Path, output_dir: Path, overwrite: bool) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / source_h5_path.name
    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_h5_path}")

    payload, num_slices = _build_bank_payload(source_h5_path)
    temp_path = output_h5_path.with_name(f".{output_h5_path.name}.state_bank_tmp_{os.getpid()}")

    try:
        with h5py.File(temp_path, "w") as output_hf:
            for key, value in sorted(payload.items()):
                _write_dataset(output_hf, key, value)

        _validate_output(temp_path, source_h5_path)
        temp_path.replace(output_h5_path)
        logging.info("Built state bank %s", output_h5_path.name)
        return output_h5_path, num_slices
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config with process settings.")
    parser.add_argument("--input-dir", default=None, help="Directory containing source DWI H5 files.")
    parser.add_argument("--output-dir", default=None, help="Directory where state-bank H5 files will be written.")
    parser.add_argument("--glob", default=None, help="Glob pattern for selecting source H5 files.")
    parser.add_argument("--job-index", type=int, default=0, help="Zero-based shard index.")
    parser.add_argument("--job-count", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output H5 files.")
    parser.add_argument("--manifest-path", default=None, help="Optional JSONL manifest path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir, output_dir, glob_pattern, overwrite, manifest_path = _resolve_runtime_args(args)
    h5_files = _list_h5_files(input_dir, glob_pattern)
    if not h5_files:
        logging.warning("No H5 files found in %s matching %s", input_dir, glob_pattern)
        return

    assigned_files, start, end = _select_assigned_files(h5_files, args.job_index, args.job_count)
    if not assigned_files:
        logging.info(
            "Job %d has no assigned files (job count %d, total files %d)",
            args.job_index,
            args.job_count,
            len(h5_files),
        )
        return

    logging.info(
        "Processing %d source files from indices [%d, %d) out of %d total files",
        len(assigned_files),
        start,
        end,
        len(h5_files),
    )

    success_count = 0
    skipped_count = 0
    error_count = 0

    for source_h5_path in assigned_files:
        output_h5_path = output_dir / source_h5_path.name
        row = {
            "source": str(source_h5_path),
            "target": str(output_h5_path),
            "status": None,
            "num_slices": None,
            "error": None,
        }
        try:
            built_path, num_slices = build_state_bank_h5(source_h5_path, output_dir, overwrite=overwrite)
            row["status"] = "ok"
            row["target"] = str(built_path)
            row["num_slices"] = num_slices
            success_count += 1
        except FileExistsError as exc:
            logging.info("Skipping existing output %s", output_h5_path.name)
            row["status"] = "skipped"
            row["error"] = str(exc)
            skipped_count += 1
        except (KeyError, RuntimeError, ValueError) as exc:
            logging.warning("Skipping malformed source %s: %s", source_h5_path.name, exc)
            row["status"] = "skipped"
            row["error"] = str(exc)
            skipped_count += 1
        except Exception as exc:  # noqa: BLE001
            logging.exception("Unhandled error while processing %s", source_h5_path.name)
            row["status"] = "error"
            row["error"] = repr(exc)
            error_count += 1
        finally:
            _write_manifest_row(manifest_path, row)

    logging.info(
        "Finished shard %d/%d: %d ok, %d skipped, %d errors",
        args.job_index,
        args.job_count,
        success_count,
        skipped_count,
        error_count,
    )


if __name__ == "__main__":
    main()
