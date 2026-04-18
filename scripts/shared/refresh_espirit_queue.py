"""Shared helpers for building the DWI ESPIRiT refresh queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REFRESH_MODALITY_DIR = "Hersh_VidaProstateDiffusion"


@dataclass(frozen=True)
class RefreshQueueEntry:
    accession_id: str
    phase: str
    target_h5_path: str
    target_h5_filename: str
    source_dat_path: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def parse_recon_h5_filename(recon_h5_filename: str) -> Tuple[str, str]:
    """Extract scanner site and base acquisition stem from a recon H5 filename."""

    stem = Path(recon_h5_filename).stem
    parts = stem.split("#")

    site = None
    for part in parts:
        if part.startswith("S") and len(part) > 1:
            site = part[1:]
            break

    if site is None:
        raise ValueError(f"Could not parse site from recon filename '{recon_h5_filename}'")

    base_stem = stem.split("__")[0]
    return site, base_stem


def derive_source_dat_path(source_root: Path, recon_h5_filename: str) -> Path:
    """Map a recon H5 filename to its source DAT path in the archive tree."""

    site, base_stem = parse_recon_h5_filename(recon_h5_filename)
    return source_root / site / REFRESH_MODALITY_DIR / f"{base_stem}.dat"


def _load_manifest_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required = {"accession_id", "recon_h5_filename", "recon_h5_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df["accession_id"] = df["accession_id"].astype(str)
    df["recon_h5_filename"] = df["recon_h5_filename"].astype(str)
    df["recon_h5_path"] = df["recon_h5_path"].astype(str)
    return df


def build_refresh_queue(
    recon_only_manifest_csv: Path,
    aligned_manifest_csv: Path,
    source_root: Path,
) -> Tuple[List[RefreshQueueEntry], Dict[str, int], List[str]]:
    """Build aligned-first refresh queue from the two cohort manifests."""

    recon_only_df = _load_manifest_rows(recon_only_manifest_csv)
    aligned_df = _load_manifest_rows(aligned_manifest_csv)

    recon_by_accession: Dict[str, pd.Series] = {}
    recon_accession_order: List[str] = []
    for _, row in recon_only_df.iterrows():
        accession_id = str(row["accession_id"])
        if accession_id in recon_by_accession:
            continue
        recon_by_accession[accession_id] = row
        recon_accession_order.append(accession_id)

    aligned_accession_order: List[str] = []
    seen_aligned: set[str] = set()
    for _, row in aligned_df.iterrows():
        accession_id = str(row["accession_id"])
        if accession_id in seen_aligned:
            continue
        aligned_accession_order.append(accession_id)
        seen_aligned.add(accession_id)

    queue: List[RefreshQueueEntry] = []
    queued_accessions: set[str] = set()
    missing_aligned: List[str] = []

    for accession_id in aligned_accession_order:
        row = recon_by_accession.get(accession_id)
        if row is None:
            missing_aligned.append(accession_id)
            continue
        queue.append(
            RefreshQueueEntry(
                accession_id=accession_id,
                phase="aligned",
                target_h5_path=str(row["recon_h5_path"]),
                target_h5_filename=str(row["recon_h5_filename"]),
                source_dat_path=str(derive_source_dat_path(source_root, str(row["recon_h5_filename"]))),
            )
        )
        queued_accessions.add(accession_id)

    for accession_id in recon_accession_order:
        if accession_id in queued_accessions:
            continue
        row = recon_by_accession[accession_id]
        queue.append(
            RefreshQueueEntry(
                accession_id=accession_id,
                phase="remaining",
                target_h5_path=str(row["recon_h5_path"]),
                target_h5_filename=str(row["recon_h5_filename"]),
                source_dat_path=str(derive_source_dat_path(source_root, str(row["recon_h5_filename"]))),
            )
        )
        queued_accessions.add(accession_id)

    counts = {
        "total_queued_count": len(queue),
        "aligned_expected_count": sum(1 for entry in queue if entry.phase == "aligned"),
        "remaining_count": sum(1 for entry in queue if entry.phase == "remaining"),
        "missing_aligned_count": len(missing_aligned),
    }
    return queue, counts, missing_aligned
