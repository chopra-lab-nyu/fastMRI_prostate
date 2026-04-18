"""Streaming worker that patches ESPIRiT datasets into existing DWI H5 files."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import yaml

from scripts.streaming.dwi.refresh_espirit_h5 import refresh_espirit_h5
from scripts.streaming.dwi.worker import acquire_lock


READY_EXT = ".dat.ready"
LOCK_EXT = ".lock"
ACTIVE_FLAG = ".transfer_active"
QUEUE_SUMMARY_FILE = "queue_summary.json"
ALIGNED_DONE_DIR = ".aligned_done"
ALIGNED_COMPLETE_MARKER = "aligned_complete.marker"


def prepare_logger(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _sidecar_path(dat_file: Path) -> Path:
    return dat_file.with_suffix(dat_file.suffix + ".meta.json")


def _failed_marker_path(dat_file: Path) -> Path:
    return dat_file.with_suffix(dat_file.suffix + ".failed")


def _mark_aligned_done(staging: Path, accession_id: str) -> None:
    done_dir = staging / ALIGNED_DONE_DIR
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / f"{accession_id}.done").touch(exist_ok=True)


def _maybe_write_aligned_complete(staging: Path) -> None:
    summary_path = staging / QUEUE_SUMMARY_FILE
    complete_marker = staging / ALIGNED_COMPLETE_MARKER
    if complete_marker.exists() or not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text())
    expected = int(summary.get("aligned_expected_count", 0))
    if expected <= 0:
        try:
            with complete_marker.open("x") as f:
                f.write("No aligned accessions queued.\n")
        except FileExistsError:
            pass
        return

    done_dir = staging / ALIGNED_DONE_DIR
    done_count = len(list(done_dir.glob("*.done")))
    if done_count < expected:
        return

    try:
        with complete_marker.open("x") as f:
            f.write(f"Aligned refresh complete: {done_count}/{expected}\n")
        logging.info("Aligned refresh complete: %d/%d", done_count, expected)
    except FileExistsError:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/streaming/refresh_espirit.yaml")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_logger(args.log_level.upper())

    cfg = yaml.safe_load(Path(args.config).read_text())
    staging = Path(cfg["transfer"]["staging_dir"])
    process_cfg = cfg["process"]

    enable_phasecorr = bool(process_cfg.get("enable_phasecorr", False))
    delete_dat = bool(process_cfg["delete_dat"])
    poll_seconds = int(process_cfg["poll_seconds"])

    logging.info("Refresh worker %d ready (staging=%s)", args.worker_id, staging)

    while True:
        ready_files = list(staging.glob(f"*{READY_EXT}"))
        random.shuffle(ready_files)
        claimed = False

        for marker in ready_files:
            dat_file = marker.with_suffix("")
            lock_dir = acquire_lock(dat_file)
            if lock_dir is None:
                continue

            claimed = True
            sidecar_path = _sidecar_path(dat_file)
            failed_marker = _failed_marker_path(dat_file)
            try:
                if not sidecar_path.exists():
                    raise FileNotFoundError(f"Missing sidecar metadata for {dat_file.name}")

                metadata = json.loads(sidecar_path.read_text())
                accession_id = str(metadata["accession_id"])
                phase = str(metadata["phase"])
                target_h5_path = Path(metadata["target_h5_path"])

                logging.info(
                    "Worker %d refreshing accession=%s phase=%s dat=%s target=%s",
                    args.worker_id,
                    accession_id,
                    phase,
                    dat_file.name,
                    target_h5_path.name,
                )
                refresh_espirit_h5(dat_file, target_h5_path, enable_phasecorr=enable_phasecorr)

                failed_marker.unlink(missing_ok=True)
                if phase == "aligned":
                    _mark_aligned_done(staging, accession_id)
                    _maybe_write_aligned_complete(staging)

                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                    sidecar_path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                logging.info("Completed accession=%s", accession_id)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed refresh for %s", dat_file.name)
                failed_marker.write_text(f"{type(exc).__name__}: {exc}\n")
                marker.unlink(missing_ok=True)
            finally:
                lock_dir.rmdir()
            break

        if not claimed:
            active = (staging / ACTIVE_FLAG).exists()
            if not active and not ready_files:
                logging.info("Refresh worker %d exiting (no files, transfer inactive)", args.worker_id)
                break
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
