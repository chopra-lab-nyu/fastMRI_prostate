"""Plot central slices of ADC and b1500 maps for each coil combine in a recon H5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _decode_strings(values: Iterable[object]) -> List[str]:
    decoded: List[str] = []
    for val in values:
        if isinstance(val, bytes):
            decoded.append(val.decode(errors="ignore"))
        else:
            decoded.append(str(val))
    return decoded


def _pick_slice(volume: np.ndarray, slice_index: int | None) -> tuple[int, np.ndarray]:
    """Return a slice index and the corresponding 2D slice."""
    if volume.ndim == 2:
        return 0, volume

    z = volume.shape[0]
    if slice_index is None:
        slice_index = z // 2
    slice_index = max(0, min(slice_index, z - 1))
    return slice_index, volume[slice_index]


def plot_metrics_slice(h5_path: Path, tag: str | None, slice_index: int | None) -> None:
    """Plot ADC and b1500 central slice for each combine present in the file."""
    with h5py.File(h5_path, "r") as f:
        combines = _decode_strings(f["metadata/combines"][()])
        tags = _decode_strings(f["metadata/averaging_schemes"][()])
        if not tags:
            raise ValueError("No averaging schemes found in file.")
        selected_tag = tag or tags[0]
        if selected_tag not in tags:
            raise ValueError(f"Tag '{selected_tag}' not found. Available: {tags}")

        rows = len(combines)
        fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows), squeeze=False)

        for row, combine in enumerate(combines):
            adc_path = f"metrics/{combine}/{selected_tag}/adc_map"
            b1500_path = f"metrics/{combine}/{selected_tag}/b1500"

            if adc_path not in f or b1500_path not in f:
                print(f"Skipping {combine}: metrics not found for tag '{selected_tag}'")
                continue

            adc = f[adc_path][()]
            b1500 = f[b1500_path][()]
            adc_slice_idx, adc_slice = _pick_slice(adc, slice_index)
            _, b1500_slice = _pick_slice(b1500, adc_slice_idx)

            print(f"{combine}: adc shape {adc.shape}, b1500 shape {b1500.shape}, slice {adc_slice_idx}")

            adc_im = axes[row, 0].imshow(adc_slice, cmap="gray")
            axes[row, 0].set_title(f"{combine} ADC (slice {adc_slice_idx})")
            axes[row, 0].axis("off")
            fig.colorbar(adc_im, ax=axes[row, 0], fraction=0.046, pad=0.04)

            b1500_im = axes[row, 1].imshow(b1500_slice, cmap="gray")
            axes[row, 1].set_title(f"{combine} b1500 (slice {adc_slice_idx})")
            axes[row, 1].axis("off")
            fig.colorbar(b1500_im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        fig.suptitle(f"{h5_path.name} – tag {selected_tag}")
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ADC and b1500 central slice from recon H5.")
    parser.add_argument("h5_path", type=Path, help="Path to reconstructed .h5 file")
    parser.add_argument(
        "--tag",
        default=None,
        help="Averaging scheme tag to use (defaults to first in metadata/averaging_schemes)",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index to show (defaults to central slice)",
    )
    args = parser.parse_args()

    plot_metrics_slice(args.h5_path, args.tag, args.slice_index)


if __name__ == "__main__":
    main()
