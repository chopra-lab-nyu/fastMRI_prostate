"""Refresh ESPIRiT datasets inside an existing DWI reconstruction H5."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np

from fastmri_prostate.data.mri_data import load_dat_file_dwi
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import dwi_reconstruction_diffusion
from scripts.streaming.dwi.recon_from_dat import build_dwi_payload


os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


SCHEME_PATTERN = re.compile(r"^b50_(\d+)_b1000_(\d+)$")


def _decode_string_values(dataset_value: np.ndarray) -> List[str]:
    if isinstance(dataset_value, bytes):
        return [dataset_value.decode(errors="ignore")]
    if isinstance(dataset_value, np.ndarray) and dataset_value.dtype.kind == "S":
        return [item.decode(errors="ignore") for item in dataset_value.tolist()]
    return [str(item) for item in np.atleast_1d(dataset_value).tolist()]


def parse_averaging_schemes(scheme_tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    schemes: List[Tuple[str, int, int]] = []
    for tag in scheme_tags:
        match = SCHEME_PATTERN.fullmatch(tag)
        if match is None:
            raise ValueError(f"Unsupported averaging scheme tag '{tag}'")
        schemes.append((tag, int(match.group(1)), int(match.group(2))))
    return schemes


def load_target_refresh_spec(target_h5_path: Path) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    with h5py.File(target_h5_path, "r") as hf:
        if "metadata/directions" not in hf:
            raise KeyError(f"Missing metadata/directions in {target_h5_path}")
        if "metadata/averaging_schemes" not in hf:
            raise KeyError(f"Missing metadata/averaging_schemes in {target_h5_path}")

        directions = _decode_string_values(hf["metadata/directions"][()])
        scheme_tags = _decode_string_values(hf["metadata/averaging_schemes"][()])

    return directions, parse_averaging_schemes(scheme_tags)


def build_espirit_refresh_payload(
    dat_file: Path,
    target_h5_path: Path,
    enable_phasecorr: bool = False,
) -> dict[str, np.ndarray]:
    directions, averaging_schemes = load_target_refresh_spec(target_h5_path)

    phasecorr = None
    if enable_phasecorr:
        kspace, calibration, hdr, phasecorr = load_dat_file_dwi(dat_file, include_phasecorr=True)
    else:
        kspace, calibration, hdr = load_dat_file_dwi(dat_file)

    recon = dwi_reconstruction_diffusion(
        kspace,
        calibration,
        hdr,
        phasecorr=phasecorr,
        directions=directions,
        enable_esc=False,
        enable_espirit=True,
        compute_metrics=False,
        enable_phasecorr=enable_phasecorr,
    )
    if recon.espirit_images_per_average is None:
        raise RuntimeError("ESPIRiT reconstruction did not produce per-average images.")

    payload = build_dwi_payload(
        recon_result=recon,
        directions=directions,
        averaging_schemes=averaging_schemes,
        compute_metrics=True,
        combines=["espirit"],
        store_kspace=False,
    )
    return {
        key: value
        for key, value in payload.items()
        if key.startswith("images/espirit/") or key.startswith("metrics/espirit/")
    }


def _ensure_parent_group(hf: h5py.File, dataset_path: str) -> None:
    parts = dataset_path.split("/")[:-1]
    if not parts:
        return
    current = hf
    for part in parts:
        if part not in current:
            current = current.create_group(part)
        else:
            current = current[part]


def _delete_group_if_present(hf: h5py.File, group_path: str) -> None:
    if group_path in hf:
        del hf[group_path]


def _write_payload(hf: h5py.File, payload: dict[str, np.ndarray]) -> None:
    for group_path in ("images/espirit", "metrics/espirit"):
        _delete_group_if_present(hf, group_path)

    for key, value in sorted(payload.items()):
        _ensure_parent_group(hf, key)
        hf.create_dataset(key, data=value)


def _validate_payload(temp_path: Path, payload_keys: Iterable[str]) -> None:
    with h5py.File(temp_path, "r") as hf:
        missing = [key for key in payload_keys if key not in hf]
    if missing:
        raise RuntimeError(f"Missing refreshed datasets after patch: {missing[:5]}")


def refresh_espirit_h5(dat_file: Path, target_h5_path: Path, enable_phasecorr: bool = False) -> Path:
    if not dat_file.exists():
        raise FileNotFoundError(f"Source DAT not found: {dat_file}")
    if not target_h5_path.exists():
        raise FileNotFoundError(f"Target H5 not found: {target_h5_path}")

    payload = build_espirit_refresh_payload(dat_file, target_h5_path, enable_phasecorr=enable_phasecorr)
    temp_path = target_h5_path.with_name(f".{target_h5_path.name}.refresh_tmp_{os.getpid()}")

    try:
        shutil.copy2(target_h5_path, temp_path)
        with h5py.File(temp_path, "r+") as hf:
            _write_payload(hf, payload)
        _validate_payload(temp_path, payload.keys())
        temp_path.replace(target_h5_path)
        logging.info("Refreshed %s from %s", target_h5_path.name, dat_file.name)
        return target_h5_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat-file", type=Path, required=True)
    parser.add_argument("--target-h5", type=Path, required=True)
    parser.add_argument("--enable-phasecorr", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    refresh_espirit_h5(args.dat_file, args.target_h5, enable_phasecorr=args.enable_phasecorr)


if __name__ == "__main__":
    main()
