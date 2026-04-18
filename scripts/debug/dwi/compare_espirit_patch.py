"""Compare stored RSS/ESPIRiT metrics against patched ESPIRiT recon and save figures."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastmri_prostate.data.mri_data import load_dat_file_dwi
from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import (
    compute_averages,
    dwi_reconstruction_diffusion,
)
from fastmri_prostate.reconstruction.utils import center_crop_im, flip_im


PIRADS_PATTERN = re.compile(r"(?:PI-?RADS|Pirads)[:\s]*(\d)", re.IGNORECASE)
SCHEME_TO_COUNTS: Dict[str, Tuple[int, int]] = {
    "b50_1_b1000_1": (1, 1),
    "b50_1_b1000_2": (1, 2),
    "b50_1_b1000_3": (1, 3),
    "b50_2_b1000_6": (2, 6),
    "b50_4_b1000_12": (4, 12),
}
REQUIRED_DIRECTIONS: Tuple[str, ...] = ("b50x", "b50y", "b50z", "b1000x", "b1000y", "b1000z")


def extract_max_pirads(report: object) -> float:
    if isinstance(report, str):
        matches = PIRADS_PATTERN.findall(report)
        if matches:
            return float(max(int(score) for score in matches))
    return float("nan")


def parse_h5_filename(path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {"path": Path(path)}
    parts = Path(path).stem.split("#")
    for part in parts:
        if part.startswith("Site"):
            info["scanner_site"] = part[4:]
        elif part.startswith("F"):
            info["F_field"] = part[1:]
        elif part.startswith("M"):
            info["M_field"] = part[1:]
        elif re.fullmatch(r"D\d{6}", part):
            info["date"] = datetime.strptime(part[1:], "%d%m%y").date().isoformat()
        elif re.fullmatch(r"T\d{6}", part):
            info["time"] = datetime.strptime(part[1:], "%H%M%S").time().isoformat()
        elif "__" in part and part.split("__")[-1].isdigit():
            info["mrn"] = part.split("__")[-1]
        else:
            info.setdefault("modality_parts", []).append(part)

    if "date" in info and "time" in info:
        try:
            info["datetime"] = pd.Timestamp(f"{info['date']}T{info['time']}")
        except Exception:
            info["datetime"] = pd.NaT
    else:
        info["datetime"] = pd.NaT
    return info


def scan_h5_directory(h5_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(h5_root.glob("*.h5")):
        try:
            records.append(parse_h5_filename(path))
        except Exception as exc:  # noqa: PERF203
            logging.warning("Failed to parse %s: %s", path, exc)
    return records


def group_records_by_mrn(records: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        mrn = record.get("mrn")
        if not mrn:
            continue
        grouped.setdefault(str(mrn), []).append(record)
    for recs in grouped.values():
        recs.sort(key=lambda rec: rec.get("datetime") or pd.Timestamp.max)
    return grouped


def match_h5_to_accession(
    row: pd.Series,
    grouped: Dict[str, List[Dict[str, object]]],
    buffer_days: int = 7,
) -> Optional[Path]:
    patient_id = str(row["PatientID"])
    study_date = row["StudyDate"]
    if pd.isna(study_date):
        return None

    candidates = grouped.get(patient_id)
    if not candidates:
        return None

    study_norm = pd.Timestamp(study_date).normalize()
    best_idx = None
    best_delta = pd.Timedelta.max

    for idx, record in enumerate(candidates):
        rec_dt = record.get("datetime", pd.NaT)
        if pd.isna(rec_dt):
            continue
        delta = abs(rec_dt.normalize() - study_norm)
        if delta <= pd.Timedelta(days=buffer_days) and delta < best_delta:
            best_delta = delta
            best_idx = idx

    if best_idx is None:
        return None

    record = candidates.pop(best_idx)
    return Path(record["path"])


def build_accession_to_recon_h5(
    labels_csv: Path,
    h5_root: Path,
    buffer_days: int = 7,
) -> Dict[str, Path]:
    labels_df = pd.read_csv(labels_csv).copy()
    labels_df["PatientID"] = labels_df["PatientID"].astype(str)
    labels_df["AccessionNumber"] = labels_df["AccessionNumber"].astype(str)
    labels_df["StudyDate"] = pd.to_datetime(labels_df["StudyDate"], errors="coerce")
    labels_df["maxPIRADS"] = labels_df["ReportBody"].apply(extract_max_pirads)
    labels_df = labels_df.dropna(subset=["StudyDate", "maxPIRADS"])
    labels_df = labels_df.drop_duplicates(subset=["AccessionNumber"])

    h5_records = scan_h5_directory(h5_root)
    mrn_lookup = group_records_by_mrn(h5_records)

    accession_to_recon_h5: Dict[str, Path] = {}
    for _, row in labels_df.iterrows():
        h5_path = match_h5_to_accession(row, mrn_lookup, buffer_days=buffer_days)
        if h5_path is not None:
            accession_to_recon_h5[str(row["AccessionNumber"])] = h5_path

    return accession_to_recon_h5


def load_accessions(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    if args.accessions:
        values.extend(str(item) for item in args.accessions)
    if args.accessions_file is not None:
        with args.accessions_file.open() as f:
            for line in f:
                token = line.strip()
                if token:
                    values.append(token)
    if not values:
        raise ValueError("Provide --accessions and/or --accessions-file.")
    return values


def shard_items(items: Sequence[str], job_index: int, job_count: int) -> List[str]:
    if job_count <= 1:
        return list(items)
    if job_index < 0 or job_index >= job_count:
        raise ValueError(f"job_index must be in [0, {job_count}), got {job_index}")
    return [item for idx, item in enumerate(items) if idx % job_count == job_index]


def build_dat_lookup(dat_dir: Path) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for path in sorted(dat_dir.glob("*.dat")):
        lookup.setdefault(path.stem, []).append(path)
    return lookup


def resolve_dat_path(h5_path: Path, dat_lookup: Dict[str, List[Path]]) -> Path:
    target = h5_path.name.split("__")[0]
    matches = dat_lookup.get(target, [])
    if not matches:
        raise FileNotFoundError(f"No .dat file found for stem '{target}'")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple .dat files found for stem '{target}': {matches}")
    return matches[0]


def postprocess(volume: np.ndarray) -> np.ndarray:
    return center_crop_im(flip_im(volume.copy(), 0), (100, 100)).astype(np.float32)


def load_stored_metrics(h5_path: Path, scheme: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for combine in ("rss", "espirit"):
            for metric in ("adc_map", "b1500"):
                key = f"metrics/{combine}/{scheme}/{metric}"
                if key not in f:
                    raise KeyError(f"Missing dataset '{key}' in {h5_path}")
                out[f"{combine}_{metric}"] = f[key][:]
    return out


def run_patched_espirit_metrics(dat_path: Path, scheme: str, enable_phasecorr: bool) -> Dict[str, np.ndarray]:
    num_b50, num_b1000 = SCHEME_TO_COUNTS[scheme]
    kspace, calibration, hdr = load_dat_file_dwi(dat_path)
    recon = dwi_reconstruction_diffusion(
        kspace,
        calibration,
        hdr,
        enable_esc=False,
        enable_espirit=True,
        enable_phasecorr=enable_phasecorr,
        compute_metrics=False,
    )
    if recon.espirit_images_per_average is None:
        raise RuntimeError("Patched reconstruction did not produce ESPIRiT images.")

    espirit_avg = compute_averages(recon.espirit_images_per_average, num_b50, num_b1000)
    metric_input = {direction: espirit_avg[direction].copy() for direction in REQUIRED_DIRECTIONS}
    metrics = compute_trace_adc_b1500(metric_input)
    return {
        "adc_map": postprocess(metrics["adc_map"]),
        "b1500": postprocess(metrics["b1500"]),
    }


def select_slice(volume: np.ndarray, slice_index: Optional[int]) -> int:
    if volume.ndim < 3:
        return 0
    if slice_index is None:
        return volume.shape[0] // 2
    return max(0, min(slice_index, volume.shape[0] - 1))


def save_comparison_figure(
    accession: str,
    h5_path: Path,
    dat_path: Path,
    output_dir: Path,
    scheme: str,
    old_metrics: Dict[str, np.ndarray],
    patched_metrics: Dict[str, np.ndarray],
    slice_index: Optional[int],
) -> Path:
    slice_idx = select_slice(old_metrics["rss_adc_map"], slice_index)

    old_rss_adc = old_metrics["rss_adc_map"][slice_idx]
    old_esp_adc = old_metrics["espirit_adc_map"][slice_idx]
    new_esp_adc = patched_metrics["adc_map"][slice_idx]
    old_rss_b1500 = old_metrics["rss_b1500"][slice_idx]
    old_esp_b1500 = old_metrics["espirit_b1500"][slice_idx]
    new_esp_b1500 = patched_metrics["b1500"][slice_idx]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14), constrained_layout=True)

    adc_vmax = np.percentile(np.stack([old_rss_adc, old_esp_adc, new_esp_adc]), 99.5)
    b1500_vmax = np.percentile(np.stack([old_rss_b1500, old_esp_b1500, new_esp_b1500]), 99.5)

    axes[0, 0].imshow(old_rss_adc, cmap="gray", vmin=0, vmax=adc_vmax)
    axes[0, 0].set_title("Stored RSS ADC")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(old_esp_adc, cmap="gray", vmin=0, vmax=adc_vmax)
    axes[0, 1].set_title("Stored ESPIRiT ADC")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(new_esp_adc, cmap="gray", vmin=0, vmax=adc_vmax)
    axes[0, 2].set_title("Patched ESPIRiT ADC")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(old_rss_b1500, cmap="gray", vmin=0, vmax=b1500_vmax)
    axes[1, 0].set_title("Stored RSS b1500")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(old_esp_b1500, cmap="gray", vmin=0, vmax=b1500_vmax)
    axes[1, 1].set_title("Stored ESPIRiT b1500")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(new_esp_b1500, cmap="gray", vmin=0, vmax=b1500_vmax)
    axes[1, 2].set_title("Patched ESPIRiT b1500")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(old_esp_adc - old_rss_adc, cmap="bwr")
    axes[2, 0].set_title("Stored ESPIRiT ADC - RSS ADC")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(new_esp_adc - old_esp_adc, cmap="bwr")
    axes[2, 1].set_title("Patched ADC - Stored ESPIRiT ADC")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(new_esp_b1500 - old_esp_b1500, cmap="bwr")
    axes[2, 2].set_title("Patched b1500 - Stored ESPIRiT b1500")
    axes[2, 2].axis("off")

    fig.suptitle(
        f"Accession {accession}\n{h5_path.name}\n{dat_path.name}\nScheme {scheme}, slice {slice_idx}",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{accession}__{h5_path.stem}__{scheme}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compare_accession(
    accession: str,
    accession_to_h5: Dict[str, Path],
    dat_lookup: Dict[str, List[Path]],
    output_dir: Path,
    scheme: str,
    slice_index: Optional[int],
    enable_phasecorr: bool,
) -> Path:
    if accession not in accession_to_h5:
        raise KeyError(f"Accession {accession} not found in H5 mapping.")

    h5_path = accession_to_h5[accession]
    dat_path = resolve_dat_path(h5_path, dat_lookup)
    logging.info("Accession %s: h5=%s dat=%s", accession, h5_path.name, dat_path.name)

    old_metrics = load_stored_metrics(h5_path, scheme)
    patched_metrics = run_patched_espirit_metrics(dat_path, scheme, enable_phasecorr=enable_phasecorr)
    return save_comparison_figure(
        accession=accession,
        h5_path=h5_path,
        dat_path=dat_path,
        output_dir=output_dir,
        scheme=scheme,
        old_metrics=old_metrics,
        patched_metrics=patched_metrics,
        slice_index=slice_index,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True, help="Radiology labels CSV used for accession-to-H5 matching.")
    parser.add_argument("--h5-root", type=Path, required=True, help="Directory containing existing recon H5 files.")
    parser.add_argument("--dat-dir", type=Path, required=True, help="Directory containing .dat files to reconstruct.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where comparison PNGs will be written.")
    parser.add_argument("--accessions", nargs="*", default=None, help="Accession numbers to process.")
    parser.add_argument("--accessions-file", type=Path, default=None, help="Optional text file with one accession per line.")
    parser.add_argument("--scheme", choices=sorted(SCHEME_TO_COUNTS), default="b50_4_b1000_12")
    parser.add_argument("--buffer-days", type=int, default=7, help="Study-date tolerance used for accession-to-H5 matching.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice index to plot. Defaults to the middle slice.")
    parser.add_argument("--enable-phasecorr", action="store_true", help="Enable phase correction during the patched recon.")
    parser.add_argument("--job-index", type=int, default=0, help="Shard index for batch jobs.")
    parser.add_argument("--job-count", type=int, default=1, help="Number of shards for batch jobs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    accessions = shard_items(load_accessions(args), args.job_index, args.job_count)
    if not accessions:
        logging.info("No accessions assigned to shard %d/%d", args.job_index, args.job_count)
        return

    accession_to_h5 = build_accession_to_recon_h5(args.labels_csv, args.h5_root, buffer_days=args.buffer_days)
    dat_lookup = build_dat_lookup(args.dat_dir)

    logging.info("Processing %d accession(s)", len(accessions))
    written: List[Path] = []
    failures: List[Tuple[str, str]] = []
    for accession in accessions:
        try:
            out_path = compare_accession(
                accession=accession,
                accession_to_h5=accession_to_h5,
                dat_lookup=dat_lookup,
                output_dir=args.output_dir,
                scheme=args.scheme,
                slice_index=args.slice_index,
                enable_phasecorr=args.enable_phasecorr,
            )
            written.append(out_path)
            logging.info("Wrote %s", out_path)
        except Exception as exc:  # noqa: PERF203
            failures.append((accession, repr(exc)))
            logging.exception("Failed accession %s", accession)

    logging.info("Finished. Wrote %d figure(s).", len(written))
    if failures:
        logging.warning("Failures: %s", failures)


if __name__ == "__main__":
    main()
