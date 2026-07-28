"""Streaming worker for direct DWI DAT to ESPIRiT state-bank conversion."""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import yaml

from scripts.streaming.dwi.direct_state_bank_from_dat import _bank_source_stems, build_state_bank_h5_from_dat
from scripts.streaming.dwi.recon_from_dat import UnsupportedAverageCountError
from scripts.streaming.dwi.transfer import ACTIVE_FLAG


READY_EXT = ".dat.ready"
LOCK_EXT = ".lock"


def acquire_lock(dat_file: Path) -> Path | None:
    lock_dir = dat_file.with_suffix(dat_file.suffix + LOCK_EXT)
    try:
        lock_dir.mkdir()
        return lock_dir
    except FileExistsError:
        return None


def transfer_active(staging: Path) -> bool:
    return (staging / ACTIVE_FLAG).exists() or any(staging.glob(f"{ACTIVE_FLAG}.*"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/streaming/dwi_direct_state_bank.yaml")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    staging = Path(cfg["transfer"]["staging_dir"])
    process_cfg = cfg["process"]
    output_dir = Path(process_cfg["output_dir"])
    source_h5_dir = Path(process_cfg["source_h5_dir_for_metadata"]) if process_cfg.get("source_h5_dir_for_metadata") else None
    overwrite = bool(process_cfg.get("overwrite", False))
    enable_phasecorr = bool(process_cfg.get("enable_phasecorr", False))
    delete_dat = bool(process_cfg.get("delete_dat", True))
    poll_seconds = int(process_cfg.get("poll_seconds", 10))

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Worker %d ready (staging=%s, output=%s)", args.worker_id, staging, output_dir)

    existing_stems = _bank_source_stems(output_dir)
    transfer_seen = transfer_active(staging)
    while True:
        ready_files = list(staging.glob(f"*{READY_EXT}"))
        transfer_seen = transfer_seen or bool(ready_files)
        random.shuffle(ready_files)
        claimed = False

        for marker in ready_files:
            dat_file = marker.with_suffix("")
            lock_dir = acquire_lock(dat_file)
            if lock_dir is None:
                continue

            claimed = True
            try:
                if dat_file.stem in existing_stems and not overwrite:
                    logging.info("Skipping already banked DAT stem %s", dat_file.stem)
                    if delete_dat:
                        dat_file.unlink(missing_ok=True)
                    marker.unlink(missing_ok=True)
                    break

                output_path, num_slices = build_state_bank_h5_from_dat(
                    dat_file,
                    output_dir,
                    source_h5_dir,
                    overwrite=overwrite,
                    enable_phasecorr=enable_phasecorr,
                )
                existing_stems.add(dat_file.stem)
                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                logging.info("Completed %s -> %s slices=%d", dat_file.name, output_path.name, num_slices)
            except (FileExistsError, UnsupportedAverageCountError, ValueError) as exc:
                logging.warning("Skipping %s: %s", dat_file.name, exc)
                if delete_dat:
                    dat_file.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
            except Exception:
                logging.exception("Failed %s", dat_file.name)
                dat_file.with_suffix(dat_file.suffix + ".failed").touch(exist_ok=True)
                marker.unlink(missing_ok=True)
            finally:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
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
