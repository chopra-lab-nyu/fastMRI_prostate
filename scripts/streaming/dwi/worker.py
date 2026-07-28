"""Streaming reconstruction worker."""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import yaml

from scripts.streaming.dwi.recon_from_dat import (
    DEFAULT_AVERAGING_SCHEMES,
    DEFAULT_DIRECTIONS,
    DEFAULT_COMBINES,
    parse_average_pair,
    parse_directions,
    process_dat_file,
    UnsupportedAverageCountError,
)

READY_EXT = ".dat.ready"
LOCK_EXT = ".lock"
ACTIVE_FLAG = ".transfer_active"


def prepare_logger(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def acquire_lock(dat_file: Path) -> Path | None:
    """Attempt to acquire exclusive lock on a dat file using atomic mkdir."""
    lock_dir = dat_file.with_suffix(dat_file.suffix + LOCK_EXT)
    try:
        lock_dir.mkdir()
        return lock_dir
    except FileExistsError:
        return None


def transfer_active(staging: Path) -> bool:
    return (staging / ACTIVE_FLAG).exists() or any(staging.glob(f"{ACTIVE_FLAG}.*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch staging directory and run ESC reconstruction")
    parser.add_argument("--config", default="config/streaming/dwi.yaml")
    parser.add_argument("--worker-id", type=int, required=True, help="Unique ID for this worker (for logging)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    prepare_logger(args.log_level.upper())

    cfg = yaml.safe_load(Path(args.config).read_text())
    staging = Path(cfg["transfer"]["staging_dir"])
    process_cfg = cfg["process"]
    output_dir = Path(process_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dirs = process_cfg.get("directions")
    if raw_dirs in (None, [], "all", "ALL"):
        directions = list(DEFAULT_DIRECTIONS)
    else:
        directions = parse_directions(",".join(raw_dirs))

    raw_avg = process_cfg.get("averages")
    if raw_avg in (None, [], "all", "ALL"):
        averaging_schemes = list(DEFAULT_AVERAGING_SCHEMES)
    else:
        avg_pair = parse_average_pair(f"{raw_avg[0]}:{raw_avg[1]}")
        averaging_schemes = [(f"b50_{avg_pair[0]}_b1000_{avg_pair[1]}", avg_pair[0], avg_pair[1])]

    raw_combines = process_cfg.get("combines")
    if raw_combines in (None, [], "all", "ALL"):
        combines = list(DEFAULT_COMBINES)
    else:
        combines = [c.strip() for c in raw_combines if c.strip()]

    skip_metrics = bool(process_cfg["skip_metrics"])
    skip_kspace = bool(process_cfg.get("skip_kspace", False))
    enable_phasecorr = bool(process_cfg.get("enable_phasecorr", False))
    delete_dat = bool(process_cfg["delete_dat"])
    poll_seconds = int(process_cfg["poll_seconds"])

    logging.info(
        "Worker %d ready (staging=%s, output=%s)",
        args.worker_id,
        staging,
        output_dir,
    )

    transfer_seen = transfer_active(staging)
    while True:
        ready_files = list(staging.glob(f"*{READY_EXT}"))
        transfer_seen = transfer_seen or bool(ready_files)
        random.shuffle(ready_files)  # Reduce lock contention across workers
        claimed = False

        for marker in ready_files:
            dat_file = marker.with_suffix("")  # drop .ready
            # Work-stealing: any worker can grab any unlocked file
            lock_dir = acquire_lock(dat_file)
            if lock_dir is None:
                continue

            claimed = True
            logging.info("Processing %s", dat_file.name)
            try:
                result_path = process_dat_file(
                    dat_file=dat_file,
                    directions=directions,
                    averaging_schemes=averaging_schemes,
                    output_dir=output_dir,
                    skip_metrics=skip_metrics,
                    combines=combines,
                    enable_phasecorr=enable_phasecorr,
                    store_kspace=not skip_kspace,
                )
            except UnsupportedAverageCountError as exc:
                logging.warning("%s; removing ready marker", exc)
                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
            except Exception:
                logging.exception("Failed %s", dat_file.name)
                failed_marker = dat_file.with_suffix(dat_file.suffix + ".failed")
                failed_marker.touch(exist_ok=True)
                marker.unlink(missing_ok=True)
            else:
                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                logging.info("Completed %s -> %s", dat_file.name, result_path)
            finally:
                lock_dir.rmdir()
            break

        if not claimed:
            active = transfer_active(staging)
            transfer_seen = transfer_seen or active
            if transfer_seen and not active and not ready_files:
                logging.info("Worker %d exiting (no files, transfer inactive)", args.worker_id)
                break
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
