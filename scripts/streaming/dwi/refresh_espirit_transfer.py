"""Stage DAT files for the DWI ESPIRiT refresh pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from scripts.shared.refresh_espirit_queue import build_refresh_queue
from scripts.streaming.dwi.transfer import copy_file, throttle


ACTIVE_FLAG = ".transfer_active"
READY_EXT = ".dat.ready"
SIDECAR_EXT = ".dat.meta.json"
STATE_FILE = ".copied_accessions.json"
QUEUE_SUMMARY_FILE = "queue_summary.json"
ALIGNED_DONE_DIR = ".aligned_done"
ALIGNED_COMPLETE_MARKER = "aligned_complete.marker"


def _sidecar_path(dat_file: Path) -> Path:
    return dat_file.with_suffix(SIDECAR_EXT)


def _write_queue_summary(
    staging_dir: Path,
    counts: dict[str, int],
    missing_aligned: list[str],
) -> None:
    payload = dict(counts)
    payload["missing_aligned_accessions"] = missing_aligned
    (staging_dir / QUEUE_SUMMARY_FILE).write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/streaming/refresh_espirit.yaml")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    transfer_cfg = cfg["transfer"]

    source_root = Path(transfer_cfg["source_root"])
    recon_only_manifest_csv = Path(transfer_cfg["recon_only_manifest_csv"])
    aligned_manifest_csv = Path(transfer_cfg["aligned_manifest_csv"])
    staging_dir = Path(transfer_cfg["staging_dir"])
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / ALIGNED_DONE_DIR).mkdir(parents=True, exist_ok=True)

    queue, counts, missing_aligned = build_refresh_queue(
        recon_only_manifest_csv=recon_only_manifest_csv,
        aligned_manifest_csv=aligned_manifest_csv,
        source_root=source_root,
    )
    _write_queue_summary(staging_dir, counts, missing_aligned)
    if counts["aligned_expected_count"] == 0:
        (staging_dir / ALIGNED_COMPLETE_MARKER).write_text("No aligned accessions queued.\n")

    max_bytes = float(transfer_cfg["max_staging_gb"]) * 1e9
    poll_seconds = int(transfer_cfg["poll_seconds"])

    (staging_dir / ACTIVE_FLAG).touch()
    state_path = staging_dir / STATE_FILE
    copied = set(json.loads(state_path.read_text())["accessions"]) if state_path.exists() else set()

    logging.info(
        "Refresh queue: total=%d aligned=%d remaining=%d missing_aligned=%d",
        counts["total_queued_count"],
        counts["aligned_expected_count"],
        counts["remaining_count"],
        counts["missing_aligned_count"],
    )
    if missing_aligned:
        logging.warning("Aligned accessions missing from recon_only queue: %s", missing_aligned[:10])

    try:
        for entry in queue:
            accession_id = entry.accession_id
            if accession_id in copied:
                continue

            src = Path(entry.source_dat_path)
            if not src.exists():
                logging.warning("Missing source DAT for accession %s: %s", accession_id, src)
                continue

            throttle(staging_dir, max_bytes, poll_seconds)

            dest = staging_dir / src.name
            ready_marker = dest.with_suffix(dest.suffix + ".ready")
            sidecar_path = _sidecar_path(dest)

            logging.info(
                "Staging accession=%s phase=%s src=%s",
                accession_id,
                entry.phase,
                src,
            )
            copy_file(src, dest)
            sidecar_path.write_text(json.dumps(entry.to_dict(), indent=2))
            ready_marker.touch()

            copied.add(accession_id)
            state_path.write_text(json.dumps({"accessions": sorted(copied)}, indent=2))
    finally:
        (staging_dir / ACTIVE_FLAG).unlink(missing_ok=True)

    logging.info("Refresh transfer complete: staged %d accessions", len(copied))


if __name__ == "__main__":
    main()
