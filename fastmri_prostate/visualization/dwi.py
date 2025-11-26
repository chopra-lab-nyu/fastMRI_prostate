"""DWI visualization helpers."""

from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_dwi_esc_h5(
    h5_path: Union[str, Path],
    slice_idx: int = 16,
    avg_idx: int = 0,
    figsize_per_col: float = 4.0,
) -> None:
    """Display all image types from an ESC DWI HDF5 file in a subplot grid.

    Parameters
    ----------
    h5_path : str or Path
        Path to the .h5 file produced by the ESC reconstruction pipeline.
    slice_idx : int
        Slice index to display for 3D volumes.
    avg_idx : int
        Average index to display for per-average volumes.
    figsize_per_col : float
        Figure width per column in inches.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as hf:
        panels: list[tuple[str, np.ndarray]] = []

        # Averaged images
        if "images/averaged" in hf:
            for direction in sorted(hf["images/averaged"].keys()):
                data = hf[f"images/averaged/{direction}"][()]
                panels.append((f"avg {direction}", data[slice_idx]))

        # Per-average images (show one average)
        if "images/per_average" in hf:
            for direction in sorted(hf["images/per_average"].keys()):
                data = hf[f"images/per_average/{direction}"][()]
                panels.append((f"per_avg {direction}[{avg_idx}]", data[avg_idx, slice_idx]))

        # ESC full (mean across averages)
        if "images/esc_full" in hf:
            esc_full = hf["images/esc_full"][()]
            panels.append(("esc_full (mean)", esc_full[:, slice_idx].mean(axis=0)))

        # Metrics (if present)
        if "metrics" in hf:
            for key in sorted(hf["metrics"].keys()):
                data = hf[f"metrics/{key}"][()]
                panels.append((f"metrics/{key}", data[slice_idx]))

        # K-space magnitude (log scale)
        if "kspace/esc" in hf:
            for direction in sorted(hf["kspace/esc"].keys()):
                kdata = hf[f"kspace/esc/{direction}"][()]
                panels.append((f"|k| {direction}", np.log1p(np.abs(kdata[avg_idx, slice_idx]))))

    n = len(panels)
    if n == 0:
        print(f"No displayable datasets found in {h5_path}")
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_col * cols, figsize_per_col * rows))
    axes = np.atleast_2d(axes).ravel()

    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{h5_path.name} | slice {slice_idx}", fontsize=12)
    plt.tight_layout()
    plt.show()
