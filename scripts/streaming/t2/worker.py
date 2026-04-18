"""Streaming T2 reconstruction worker."""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import yaml

from scripts.streaming.t2.recon_from_dat import (
    DEFAULT_AVERAGING_SCHEMES,
    UnsupportedAverageCountError,
    parse_average_schemes,
    process_dat_file,
)

READY_EXT = ".dat.ready"
LOCK_EXT = ".lock"
ACTIVE_FLAG = ".transfer_active"


def prepare_logger(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def acquire_lock(dat_file: Path) -> Path | None:
    lock_dir = dat_file.with_suffix(dat_file.suffix + LOCK_EXT)
    try:
        lock_dir.mkdir()
        return lock_dir
    except FileExistsError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch T2 staging directory and run reconstruction")
    parser.add_argument("--config", default="config/streaming/t2.yaml")
    parser.add_argument("--worker-id", type=int, required=True, help="Unique ID for this worker (for logging)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    prepare_logger(args.log_level.upper())

    cfg = yaml.safe_load(Path(args.config).read_text())
    staging = Path(cfg["transfer"]["staging_dir"])
    process_cfg = cfg["process"]
    output_dir = Path(process_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_single_average_raw = process_cfg.get("output_dir_single_average")
    output_dir_single_average = Path(output_dir_single_average_raw) if output_dir_single_average_raw else None
    if output_dir_single_average is not None:
        output_dir_single_average.mkdir(parents=True, exist_ok=True)

    raw_avg = process_cfg.get("averages")
    averaging_schemes = list(DEFAULT_AVERAGING_SCHEMES) if raw_avg in (None, [], "all", "ALL") else parse_average_schemes(raw_avg)

    skip_kspace = bool(process_cfg.get("skip_kspace", False))
    delete_dat = bool(process_cfg["delete_dat"])
    poll_seconds = int(process_cfg["poll_seconds"])

    logging.info(
        "Worker %d ready (staging=%s, output=%s)",
        args.worker_id,
        staging,
        output_dir,
    )

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
            logging.info("Processing %s", dat_file.name)
            try:
                result_path = process_dat_file(
                    dat_file=dat_file,
                    averaging_schemes=averaging_schemes,
                    output_dir=output_dir,
                    single_average_output_dir=output_dir_single_average,
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
            active = (staging / ACTIVE_FLAG).exists()
            if not active and not ready_files:
                logging.info("Worker %d exiting (no files, transfer inactive)", args.worker_id)
                break
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
