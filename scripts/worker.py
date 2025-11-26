"""Streaming reconstruction worker."""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import yaml

from fastmri_prostate_recon_from_dat import (
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
    lock_dir = dat_file.with_suffix(dat_file.suffix + LOCK_EXT)
    try:
        lock_dir.mkdir()
        return lock_dir
    except FileExistsError:
        return None


def shard_owner(name: str, worker_count: int) -> int:
    digest = hashlib.md5(name.encode("utf-8"))
    return int(digest.hexdigest(), 16) % worker_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch staging directory and run ESC reconstruction")
    parser.add_argument("--config", default="config/streaming.yaml")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    prepare_logger(args.log_level.upper())

    cfg = yaml.safe_load(Path(args.config).read_text())
    staging = Path(cfg["transfer"]["staging_dir"])
    process_cfg = cfg["process"]
    output_dir = Path(process_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    directions = parse_directions(",".join(process_cfg["directions"]))
    avg_pair = parse_average_pair(f"{process_cfg['averages'][0]}:{process_cfg['averages'][1]}")
    skip_metrics = bool(process_cfg["skip_metrics"])
    delete_dat = bool(process_cfg["delete_dat"])
    poll_seconds = int(process_cfg["poll_seconds"])

    logging.info(
        "Worker %d/%d ready (staging=%s, output=%s)",
        args.worker_id,
        args.worker_count,
        staging,
        output_dir,
    )

    while True:
        ready_files = sorted(staging.glob(f"*{READY_EXT}"))
        claimed = False

        for marker in ready_files:
            dat_file = marker.with_suffix("")  # drop .ready
            if shard_owner(dat_file.name, args.worker_count) != args.worker_id:
                continue
            lock_dir = acquire_lock(dat_file)
            if lock_dir is None:
                continue

            claimed = True
            logging.info("Processing %s", dat_file.name)
            try:
                result_path = process_dat_file(
                    dat_file=dat_file,
                    directions=directions,
                    averages=avg_pair,
                    output_dir=output_dir,
                    skip_metrics=skip_metrics,
                )
            except UnsupportedAverageCountError as exc:
                logging.warning("%s; removing ready marker", exc)
                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
            except Exception:
                logging.exception("Failed %s", dat_file.name)
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
