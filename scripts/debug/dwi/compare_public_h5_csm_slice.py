"""Fast one-slice comparison of stored H5 coil maps against candidate ESPIRiT maps."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim

from fastmri_prostate.reconstruction.dwi.coil_combine import combine_with_maps, espirit
from fastmri_prostate.reconstruction.dwi.regridding import trapezoidal_regridding as td_esc_regridding
from fastmri_prostate.reconstruction.grappa import Grappa
from fastmri_prostate.reconstruction.utils import ifftnd
from scipy.ndimage import zoom

from scripts.debug.dwi.compare_public_h5_dwi_recon import parse_public_regridding_params


@dataclass
class CandidateResult:
    name: str
    subspace_corr_mean: float
    subspace_corr_masked_mean: float
    subspace_corr_min: float
    abs_diff_mean: float
    abs_diff_max: float
    aligned_abs_diff_mean: float
    aligned_abs_diff_masked_mean: float
    aligned_abs_diff_max: float
    norm_ratio_mean: float
    norm_ratio_std: float
    image_abs_diff_mean: float
    image_abs_diff_max: float
    image_ssim: float
    candidate: np.ndarray


def resize_real_imag(maps: np.ndarray, target_x: int, target_y: int) -> np.ndarray:
    if maps.shape[1:] == (target_x, target_y):
        return maps
    zoom_factors = (1.0, target_x / maps.shape[1], target_y / maps.shape[2])
    real = zoom(maps.real, zoom_factors, order=1, mode="nearest")
    imag = zoom(maps.imag, zoom_factors, order=1, mode="nearest")
    return real + 1j * imag


def resize_mag_phase(maps: np.ndarray, target_x: int, target_y: int) -> np.ndarray:
    if maps.shape[1:] == (target_x, target_y):
        return maps
    zoom_factors = (1.0, target_x / maps.shape[1], target_y / maps.shape[2])
    mag = zoom(np.abs(maps), zoom_factors, order=1, mode="nearest")
    phase = zoom(np.angle(maps), zoom_factors, order=0, mode="nearest")
    return mag * np.exp(1j * phase)


def normalize_per_pixel(maps: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return maps / (scale + eps)


def mean_over_mask(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float(np.mean(values))
    if values.ndim == mask.ndim:
        return float(np.mean(values[mask]))
    if values.ndim == mask.ndim + 1:
        return float(np.mean(values[:, mask]))
    raise ValueError(f"Unsupported values shape {values.shape} for mask shape {mask.shape}")


def subspace_correlation(reference: np.ndarray, candidate: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.abs(np.sum(np.conj(reference) * candidate, axis=0))
    denominator = np.sqrt(
        np.sum(np.abs(reference) ** 2, axis=0) * np.sum(np.abs(candidate) ** 2, axis=0)
    ) + eps
    return numerator / denominator


def align_candidate_to_reference(reference: np.ndarray, candidate: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    alpha = np.sum(np.conj(candidate) * reference, axis=0) / (np.sum(np.abs(candidate) ** 2, axis=0) + eps)
    return candidate * alpha[None, ...]


def summarize_candidate(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> CandidateResult:
    corr = subspace_correlation(reference, candidate)
    aligned = align_candidate_to_reference(reference, candidate)
    diff = np.abs(candidate - reference)
    aligned_diff = np.abs(aligned - reference)

    ref_norm = np.sqrt(np.sum(np.abs(reference) ** 2, axis=0))
    cand_norm = np.sqrt(np.sum(np.abs(candidate) ** 2, axis=0))
    norm_ratio = cand_norm / (ref_norm + 1e-12)

    return CandidateResult(
        name=name,
        subspace_corr_mean=float(np.mean(corr)),
        subspace_corr_masked_mean=mean_over_mask(corr, mask),
        subspace_corr_min=float(np.min(corr)),
        abs_diff_mean=float(np.mean(diff)),
        abs_diff_max=float(np.max(diff)),
        aligned_abs_diff_mean=float(np.mean(aligned_diff)),
        aligned_abs_diff_masked_mean=mean_over_mask(aligned_diff, mask),
        aligned_abs_diff_max=float(np.max(aligned_diff)),
        norm_ratio_mean=float(np.mean(norm_ratio)),
        norm_ratio_std=float(np.std(norm_ratio)),
        image_abs_diff_mean=float("nan"),
        image_abs_diff_max=float("nan"),
        image_ssim=float("nan"),
        candidate=candidate,
    )


def build_candidates(raw_sets: np.ndarray, target_x: int, target_y: int, max_sets: int = 1) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    num_sets = min(raw_sets.shape[-1], max_sets)

    for set_idx in range(num_sets):
        lowres = raw_sets[:, :, :, set_idx].transpose(2, 0, 1)  # (coils, x, y)
        variants = {
            "ri": resize_real_imag(lowres, target_x, target_y),
            "magphase": resize_mag_phase(lowres, target_x, target_y),
        }

        for base_name, maps in variants.items():
            candidates.append((f"set{set_idx:02d}_{base_name}", maps))
            candidates.append((f"set{set_idx:02d}_{base_name}_unitnorm", normalize_per_pixel(maps)))

    return candidates


def rank_candidates(
    reference: np.ndarray,
    candidates: Sequence[Tuple[str, np.ndarray]],
    coil_images: np.ndarray,
    reference_image: np.ndarray,
) -> List[CandidateResult]:
    mask = np.sqrt(np.sum(np.abs(reference) ** 2, axis=0)) > (1e-3 * np.max(np.sqrt(np.sum(np.abs(reference) ** 2, axis=0))))
    results = [
        summarize_candidate(name, reference, candidate, mask)
        for name, candidate in candidates
    ]
    for result in results:
        candidate_image = combine_with_maps(coil_images, result.candidate)
        image_abs_diff = np.abs(candidate_image - reference_image)
        data_range = float(max(np.ptp(reference_image), np.ptp(candidate_image), 1e-8))
        result.image_abs_diff_mean = float(np.mean(image_abs_diff))
        result.image_abs_diff_max = float(np.max(image_abs_diff))
        result.image_ssim = float(ssim(reference_image, candidate_image, data_range=data_range))
    results.sort(key=lambda result: (-result.image_ssim, result.image_abs_diff_mean, -result.subspace_corr_masked_mean))
    return results


def choose_display_coil(reference: np.ndarray) -> int:
    coil_power = np.sum(np.abs(reference) ** 2, axis=(1, 2))
    return int(np.argmax(coil_power))


def save_figure(
    reference: np.ndarray,
    ranked: Sequence[CandidateResult],
    output_path: Path,
    title: str,
    top_k: int = 4,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = ranked[:top_k]
    coil_idx = choose_display_coil(reference)

    num_rows = 1 + len(selected)
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 3.8 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    ref_mag = np.abs(reference[coil_idx])
    ref_phase = np.angle(reference[coil_idx])
    ref_norm = np.sqrt(np.sum(np.abs(reference) ** 2, axis=0))

    axes[0, 0].imshow(ref_mag, cmap="gray")
    axes[0, 0].set_title(f"Stored coil {coil_idx} mag")
    axes[0, 1].imshow(ref_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title(f"Stored coil {coil_idx} phase")
    axes[0, 2].imshow(ref_norm, cmap="gray")
    axes[0, 2].set_title("Stored pixel norm")
    axes[0, 3].axis("off")

    for row_idx, result in enumerate(selected, start=1):
        cand_mag = np.abs(result.candidate[coil_idx])
        cand_phase = np.angle(result.candidate[coil_idx])
        cand_norm = np.sqrt(np.sum(np.abs(result.candidate) ** 2, axis=0))
        corr = subspace_correlation(reference, result.candidate)

        axes[row_idx, 0].imshow(cand_mag, cmap="gray")
        axes[row_idx, 0].set_title(f"{result.name} mag")
        axes[row_idx, 1].imshow(cand_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[row_idx, 1].set_title(f"{result.name} phase")
        axes[row_idx, 2].imshow(np.abs(cand_norm - ref_norm), cmap="magma")
        axes[row_idx, 2].set_title("pixel-norm abs diff")
        axes[row_idx, 3].imshow(corr, cmap="viridis", vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"corr mean={result.subspace_corr_mean:.4f}")

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(output_path: Path, ranked: Sequence[CandidateResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "subspace_corr_mean",
        "subspace_corr_masked_mean",
        "subspace_corr_min",
        "abs_diff_mean",
        "abs_diff_max",
        "aligned_abs_diff_mean",
        "aligned_abs_diff_masked_mean",
        "aligned_abs_diff_max",
        "norm_ratio_mean",
        "norm_ratio_std",
        "image_abs_diff_mean",
        "image_abs_diff_max",
        "image_ssim",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in ranked:
            writer.writerow(
                {
                    "name": result.name,
                    "subspace_corr_mean": result.subspace_corr_mean,
                    "subspace_corr_masked_mean": result.subspace_corr_masked_mean,
                    "subspace_corr_min": result.subspace_corr_min,
                    "abs_diff_mean": result.abs_diff_mean,
                    "abs_diff_max": result.abs_diff_max,
                    "aligned_abs_diff_mean": result.aligned_abs_diff_mean,
                    "aligned_abs_diff_masked_mean": result.aligned_abs_diff_masked_mean,
                    "aligned_abs_diff_max": result.aligned_abs_diff_max,
                    "norm_ratio_mean": result.norm_ratio_mean,
                    "norm_ratio_std": result.norm_ratio_std,
                    "image_abs_diff_mean": result.image_abs_diff_mean,
                    "image_abs_diff_max": result.image_abs_diff_max,
                    "image_ssim": result.image_ssim,
                }
            )


def build_espirit_inputs(calib: np.ndarray) -> List[Tuple[str, np.ndarray, bool]]:
    xyc = calib.transpose(1, 2, 0)
    return [
        ("1xyc", xyc[np.newaxis, ...], False),
        ("1yxc", xyc.transpose(1, 0, 2)[np.newaxis, ...], True),
    ]


def extract_raw_sets(esp_maps: np.ndarray, swap_back: bool) -> np.ndarray:
    raw_sets = esp_maps[0]
    if swap_back:
        raw_sets = raw_sets.transpose(1, 0, 2, 3)
    return raw_sets


def build_single_slice_coil_images(
    kspace_slice: np.ndarray,
    calibration_slice: np.ndarray,
    hdr: Dict[str, Any],
) -> np.ndarray:
    kspace_regridded = td_esc_regridding(kspace_slice, hdr)
    calib_regridded = td_esc_regridding(calibration_slice, hdr)
    kspace_for_grappa = np.transpose(kspace_regridded, (2, 0, 1))
    calib_for_grappa = np.transpose(calib_regridded, (2, 0, 1))

    grappa_obj = Grappa(kspace_for_grappa, kernel_size=(5, 5), coil_axis=1)
    weights = grappa_obj.compute_weights(calib_for_grappa)
    kspace_post_grappa = grappa_obj.apply_weights(kspace_for_grappa, weights)
    coil_domain_phase_coil_readout = ifftnd(kspace_post_grappa, [0, 2])
    return np.transpose(coil_domain_phase_coil_readout, (1, 2, 0))


def run_experiment(h5_path: Path, slice_index: int, output_dir: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        hdr = parse_public_regridding_params(f["ismrmrd_header"][()])
        kspace_slice = f["kspace"][0, slice_index]
        calibration_slice = f["calibration_data"][slice_index]
        stored_maps = f["coil_sens_maps"][slice_index]
    coil_images = build_single_slice_coil_images(kspace_slice, calibration_slice, hdr)
    reference_image = combine_with_maps(coil_images, stored_maps)

    source_variants = {
        "no_regrid": calibration_slice,
        "tdesc_regrid": td_esc_regridding(calibration_slice, hdr),
    }

    all_ranked: List[CandidateResult] = []
    for source_name, calib in source_variants.items():
        for embed_name, kspace_4d, swap_back in build_espirit_inputs(calib):
            raw_sets = extract_raw_sets(
                espirit(kspace_4d, 6, min(32, kspace_4d.shape[1], kspace_4d.shape[2]), 0.02, 0.9),
                swap_back=swap_back,
            )
            candidates = build_candidates(raw_sets, stored_maps.shape[1], stored_maps.shape[2], max_sets=1)
            ranked = rank_candidates(
                stored_maps,
                [(f"{source_name}_{embed_name}_{name}", candidate) for name, candidate in candidates],
                coil_images,
                reference_image,
            )
            all_ranked.extend(ranked)

    all_ranked.sort(key=lambda result: (-result.image_ssim, result.image_abs_diff_mean, -result.subspace_corr_masked_mean))

    csv_path = output_dir / "summary.csv"
    fig_path = output_dir / "top_candidates.png"
    title = f"{h5_path.name} | slice {slice_index}"
    write_csv(csv_path, all_ranked)
    save_figure(stored_maps, all_ranked, fig_path, title)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    print("Top candidates:")
    for result in all_ranked[:10]:
        print(
            f"{result.name}: image_ssim={result.image_ssim:.6f}, "
            f"image_abs_diff_mean={result.image_abs_diff_mean:.6f}, "
            f"corr_masked_mean={result.subspace_corr_masked_mean:.6f}, "
            f"aligned_abs_diff_masked_mean={result.aligned_abs_diff_masked_mean:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True, help="Public H5 file to inspect.")
    parser.add_argument("--slice-index", type=int, default=17, help="Slice index to analyze.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for CSV and figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.input_h5, args.slice_index, args.output_dir)


if __name__ == "__main__":
    main()
