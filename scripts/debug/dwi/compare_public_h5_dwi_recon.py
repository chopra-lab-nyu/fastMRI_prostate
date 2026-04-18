"""Compare public DWI H5 reconstructions against public and td/esc pipelines."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim


REQUIRED_DIRECTIONS: Tuple[str, ...] = ("b50x", "b50y", "b50z", "b1000x", "b1000y", "b1000z")
METRIC_NAMES: Tuple[str, ...] = ("adc_map", "b1500")
MODE_DISPLAY_NAMES: Dict[str, str] = {
    "public_csm": "Public CSM",
    "td_esc_csm": "td/esc + H5 CSM",
    "td_esc_espirit": "td/esc ESPIRiT",
}
FULL_NUM_B50_AVERAGES = 4
FULL_NUM_B1000_AVERAGES = 12


def shard_items(items: Sequence[Path], job_index: int, job_count: int) -> List[Path]:
    if job_count <= 1:
        return list(items)
    if job_index < 0 or job_index >= job_count:
        raise ValueError(f"job_index must be in [0, {job_count}), got {job_index}")
    return [item for idx, item in enumerate(items) if idx % job_count == job_index]


def postprocess(volume: np.ndarray) -> np.ndarray:
    from fastmri_prostate.reconstruction.utils import center_crop_im, flip_im

    return center_crop_im(flip_im(volume.copy(), 0), (100, 100)).astype(np.float32)


def safe_output_stem(path: Path, input_root: Optional[Path]) -> str:
    if input_root is None:
        return path.stem
    try:
        rel = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return path.stem
    return "__".join(rel.with_suffix("").parts)


def discover_h5_files(input_dir: Optional[Path], input_h5s: Optional[Sequence[Path]]) -> List[Path]:
    files: List[Path] = []
    if input_dir is not None:
        files.extend(sorted(path for path in input_dir.rglob("*.h5") if path.is_file()))
    if input_h5s:
        files.extend(Path(path) for path in input_h5s)

    deduped = sorted({path.resolve() for path in files})
    return [Path(path) for path in deduped]


def parse_public_regridding_params(hdr: Any) -> Dict[str, float]:
    if isinstance(hdr, bytes):
        hdr = hdr.decode()

    ns = {"ns": "http://www.ismrm.org/ISMRMRD"}
    root = etree.fromstring(hdr)
    params: Dict[str, float] = {}
    for node in root.findall("ns:encoding/ns:trajectoryDescription/ns:userParameterLong", ns):
        key = node[0].text
        value = node[1].text
        if key is None or value is None:
            continue
        try:
            params[key] = float(value)
        except ValueError:
            continue

    required = ("rampUpTime", "rampDownTime", "flatTopTime", "acqDelayTime")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing required regridding params in ISMRMRD header: {missing}")

    regrid_params: Dict[str, float] = {
        "rampUpTime": params["rampUpTime"],
        "rampDownTime": params["rampDownTime"],
        "flatTopTime": params["flatTopTime"],
        "acqDelayTime": params["acqDelayTime"],
        "echoSpacing": params.get("echoSpacing", params["rampUpTime"] + params["flatTopTime"] + params["rampDownTime"]),
    }

    if "destSamples" in params:
        regrid_params["destSamples"] = params["destSamples"]
    elif "numSamples" in params:
        regrid_params["destSamples"] = params["numSamples"]

    if "adcDuration" in params:
        regrid_params["adcDuration"] = params["adcDuration"]
    else:
        total_readout = regrid_params["rampUpTime"] + regrid_params["flatTopTime"] + regrid_params["rampDownTime"]
        regrid_params["adcDuration"] = total_readout - (2.0 * regrid_params["acqDelayTime"])

    return regrid_params


def load_public_dwi_h5(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        out: Dict[str, Any] = {
            "kspace": f["kspace"][:],
            "calibration": f["calibration_data"][:],
            "coil_sens_maps": f["coil_sens_maps"][:],
            "hdr": parse_public_regridding_params(f["ismrmrd_header"][()]),
            "reference": {
                "adc_map": f["adc_map"][:].astype(np.float32),
                "b1500": f["b1500"][:].astype(np.float32),
            },
        }
    return out


def public_compute_averages(img_vol: np.ndarray) -> Dict[str, np.ndarray]:
    """Literal main-branch averaging pattern used by the public DWI release."""

    return {
        "b50x": np.sum(img_vol[2:21:6, ...], axis=0) / 4,
        "b50y": np.sum(img_vol[3:22:6, ...], axis=0) / 4,
        "b50z": np.sum(img_vol[4:23:6, ...], axis=0) / 4,
        "b1000x": np.sum(
            np.r_[img_vol[5:24:6, ...], img_vol[26:48:3, ...]],
            axis=0,
        )
        / 12,
        "b1000y": np.sum(
            np.r_[img_vol[6:25:6, ...], img_vol[27:49:3, ...]],
            axis=0,
        )
        / 12,
        "b1000z": np.sum(
            np.r_[img_vol[7:26:6, ...], img_vol[28:50:3, ...]],
            axis=0,
        )
        / 12,
    }


def public_trace(img_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    trace_b50 = np.cbrt(img_dict["b50x"] * img_dict["b50y"] * img_dict["b50z"])
    trace_b1000 = np.cbrt(img_dict["b1000x"] * img_dict["b1000y"] * img_dict["b1000z"])
    return trace_b50, trace_b1000


def public_adc(raw_images: np.ndarray, adc_scale: float, b_values: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if np.mean(raw_images) < 1e-3:
        raw_images = 1e5 * raw_images

    log_image = np.log(raw_images + 1.0)
    sum_log_image = np.mean(log_image, axis=2)

    x_mat = np.column_stack((b_values, np.ones(2)))
    y_mat = sum_log_image.reshape(-1, len(b_values)).T

    res = np.linalg.lstsq(x_mat, y_mat, rcond=None)[0]
    tmp = res[0, :].reshape(sum_log_image.shape[:2])
    b0_img = np.exp(res[1, :].reshape(sum_log_image.shape[:2]))
    b0_img[np.isnan(b0_img)] = 0

    adc_map = tmp * adc_scale
    adc_map[(adc_map < 0) | np.isnan(adc_map)] = 0
    return adc_map, b0_img


def public_b1500(adc_map: np.ndarray, b0_img: np.ndarray, adc_scale: float, b_values: List[int]) -> np.ndarray:
    noise_level = 12
    noise_threshold_max_adc = 300
    calculated_b_value = 1500
    noise_threshold_min_b0 = noise_level

    minimal_pixel_fraction = 0.01
    b0_intensity = b0_img[(adc_map < noise_threshold_max_adc) & (b0_img > noise_threshold_min_b0)]
    if len(b0_intensity) > ((minimal_pixel_fraction * adc_map.size) + 1):
        noise_level = np.percentile(b0_intensity, 50) * 3

    noise_estimation_adc_offset = 1000
    adc_offset = np.where(
        (noise_level > 0) & (b0_img < noise_level),
        noise_estimation_adc_offset * np.sqrt(np.maximum(1 - ((b0_img / noise_level) ** 2), 0)),
        0,
    )

    neg_calc_b_value = calculated_b_value / adc_scale
    neg_max_b_value = b_values[-1] / adc_scale
    tmp_exponent = (neg_calc_b_value - neg_max_b_value) * np.maximum(adc_map, adc_offset) + neg_max_b_value * adc_map
    return b0_img * np.exp(tmp_exponent)


def public_compute_trace_adc_b1500(img_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Literal main-branch ADC/b1500 computation used by the public DWI release."""

    img_dict["trace_b50"], img_dict["trace_b1000"] = public_trace(img_dict)

    adc_scale = -1e6
    b_values = [50, 1000]

    recon_shape = img_dict["b50x"].shape
    adc_vol = np.zeros(shape=recon_shape + (3, 2))

    for i, b_value in enumerate([50, 1000]):
        for j, axis in enumerate(["x", "y", "z"]):
            key = f"b{b_value}{axis}"
            adc_vol[:, :, :, j, i] = img_dict[key]

    adc_map, b0_img = map(
        np.array,
        zip(*[public_adc(adc_vol[sl, ...], adc_scale, b_values) for sl in range(recon_shape[0])]),
    )

    img_dict["adc_map"] = adc_map
    img_dict["b1500"] = public_b1500(adc_map, b0_img, adc_scale, b_values)
    return img_dict


def public_get_grid_mat(epi_params: Dict[str, Any], os_factor: int, keep_oversampling: bool) -> np.ndarray:
    """Literal main-branch trapezoidal gridding matrix used for the public DWI recon."""

    t_rampup = epi_params["rampUpTime"]
    t_rampdown = epi_params["rampDownTime"]
    t_flattop = epi_params["flatTopTime"]
    t_delay = epi_params["acqDelayTime"]

    adc_nos = 200.0
    t_adcdur = 580.0

    if keep_oversampling:
        i_pts_readout = adc_nos
    else:
        i_pts_readout = adc_nos / os_factor

    if t_rampup == 0:
        return np.eye(int(i_pts_readout), int(adc_nos), dtype=np.float32)

    tt = np.linspace(t_delay, t_delay + t_adcdur, int(adc_nos))
    kk = np.zeros(shape=(int(adc_nos)))

    for zz in range(int(adc_nos)):
        if tt[zz] < t_rampup:
            kk[zz] = (0.5 / t_rampup) * np.square(tt[zz])
        elif tt[zz] > (t_rampup + t_flattop):
            kk[zz] = (
                (0.5 / t_rampup) * np.square(t_rampup)
                + (tt[zz] - t_rampup)
                - (0.5 / t_rampdown) * (np.square(tt[zz] - t_rampup - t_flattop))
            )
        else:
            kk[zz] = (0.5 / t_rampup) * np.square(t_rampup) + (tt[zz] - t_rampup)

    kk = kk - kk[int(np.floor(adc_nos / 2)) - 1]
    need_kk = np.linspace(kk[0], kk[len(kk) - 1], int(i_pts_readout))
    delta_k = need_kk[1] - need_kk[0]

    density = np.diff(kk)
    density = np.append(density, density[0])

    grid_mat = np.sinc(
        (np.tile(need_kk, (int(adc_nos), 1)).T - np.tile(kk, (int(i_pts_readout), 1))) / delta_k
    )

    grid_mat = np.tile(density, (int(i_pts_readout), 1)) * grid_mat
    grid_mat = grid_mat / (1e-12 + np.tile(np.sum(grid_mat, axis=1), (int(adc_nos), 1)).T)
    return grid_mat.astype(np.float32)


def public_trapezoidal_regridding(img: np.ndarray, epi_params: Dict[str, Any]) -> np.ndarray:
    """Literal main-branch trapezoidal regridding used for the public DWI recon."""

    grid_mat = public_get_grid_mat(epi_params, os_factor=2, keep_oversampling=True)

    img2 = np.transpose(img, (1, 2, 0))
    img_shape = img2.shape
    img2 = np.reshape(img2, (img2.shape[0], np.prod(img2.shape[1:])))

    img_out = grid_mat @ img2
    img_out = np.reshape(img_out, img_shape)

    return np.transpose(img_out, (2, 0, 1))


def reconstruct_public_csm(
    kspace: np.ndarray,
    calibration: np.ndarray,
    coil_sens_maps: np.ndarray,
    hdr: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Reproduce the public main-branch H5 reconstruction path from H5 inputs."""

    from fastmri_prostate.reconstruction.grappa import Grappa
    from fastmri_prostate.reconstruction.utils import ifftnd

    kspace_slice_regridded = public_trapezoidal_regridding(kspace[0, 0, ...], hdr)
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    grappa_weight_dict: Dict[int, Dict[Any, np.ndarray]] = {}
    for slice_num in range(kspace.shape[1]):
        calibration_regridded = public_trapezoidal_regridding(calibration[slice_num, ...], hdr)
        calib_for_grappa = np.transpose(calibration_regridded, (2, 0, 1))
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(calib_for_grappa)

    img_vol = np.zeros(
        (kspace.shape[0], kspace.shape[1], kspace.shape[3], kspace.shape[4]),
        dtype=np.float32,
    )

    for average in range(kspace.shape[0]):
        if average % 5 == 0:
            logging.info("Public-CSM recon average %d/%d", average + 1, kspace.shape[0])
        for slice_num in range(kspace.shape[1]):
            kspace_slice_regridded = public_trapezoidal_regridding(kspace[average, slice_num, ...], hdr)
            kspace_for_grappa = np.transpose(kspace_slice_regridded, (2, 0, 1))

            kspace_post_grappa = grappa_obj.apply_weights(
                kspace_for_grappa,
                grappa_weight_dict[slice_num],
            )
            img = ifftnd(kspace_post_grappa, [0, -1])
            coil_domain = np.transpose(img, (1, 2, 0))
            coil_combined = np.sum(coil_domain * coil_sens_maps[slice_num].conj(), axis=0)
            img_vol[average, slice_num] = np.abs(coil_combined).astype(np.float32, copy=False)

    img_dict = public_compute_averages(img_vol)
    metrics = public_compute_trace_adc_b1500({direction: img_dict[direction].copy() for direction in REQUIRED_DIRECTIONS})
    return {metric_name: postprocess(metrics[metric_name]) for metric_name in METRIC_NAMES}


def reconstruct_td_esc_variants(
    kspace: np.ndarray,
    calibration: np.ndarray,
    coil_sens_maps: np.ndarray,
    hdr: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the current td/esc DWI reconstruction path and expose both coil-combine variants."""

    from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
    from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import compute_averages, dwi_reconstruction_diffusion

    recon = dwi_reconstruction_diffusion(
        kspace,
        calibration,
        hdr,
        num_b50_averages=FULL_NUM_B50_AVERAGES,
        num_b1000_averages=FULL_NUM_B1000_AVERAGES,
        compute_metrics=False,
        enable_esc=False,
        enable_espirit=True,
    )
    if recon.espirit_images_per_average is None:
        raise RuntimeError("td/esc reconstruction did not return ESPIRiT images.")

    h5_csm_images = np.abs(
        np.sum(
            recon.post_grappa_coil_images * coil_sens_maps[None, ...].conj(),
            axis=2,
        )
    ).astype(np.float32, copy=False)

    variant_inputs = {
        "td_esc_csm": h5_csm_images,
        "td_esc_espirit": recon.espirit_images_per_average.astype(np.float32, copy=False),
    }

    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for mode_name, img_per_average in variant_inputs.items():
        averaged = compute_averages(img_per_average, FULL_NUM_B50_AVERAGES, FULL_NUM_B1000_AVERAGES)
        current_metrics = compute_trace_adc_b1500({direction: averaged[direction].copy() for direction in REQUIRED_DIRECTIONS})
        outputs[mode_name] = {metric_name: postprocess(current_metrics[metric_name]) for metric_name in METRIC_NAMES}

    return outputs


def compute_ssim_maps(
    reference_volume: np.ndarray,
    predicted_volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scores: List[float] = []
    maps: List[np.ndarray] = []
    for slice_num in range(reference_volume.shape[0]):
        data_range = float(max(np.ptp(reference_volume[slice_num]), np.ptp(predicted_volume[slice_num]), 1e-8))
        score, score_map = ssim(
            reference_volume[slice_num],
            predicted_volume[slice_num],
            data_range=data_range,
            full=True,
        )
        scores.append(float(score))
        maps.append(score_map.astype(np.float32))
    return np.asarray(scores, dtype=np.float32), np.stack(maps, axis=0)


def summarize_metrics(
    reference: Dict[str, np.ndarray],
    predicted: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    adc_scores, adc_maps = compute_ssim_maps(reference["adc_map"], predicted["adc_map"])
    b1500_scores, b1500_maps = compute_ssim_maps(reference["b1500"], predicted["b1500"])

    summary = {
        "adc_mean_abs": float(np.mean(np.abs(predicted["adc_map"] - reference["adc_map"]))),
        "adc_max_abs": float(np.max(np.abs(predicted["adc_map"] - reference["adc_map"]))),
        "adc_mean_ssim": float(adc_scores.mean()),
        "adc_min_ssim": float(adc_scores.min()),
        "b1500_mean_abs": float(np.mean(np.abs(predicted["b1500"] - reference["b1500"]))),
        "b1500_max_abs": float(np.max(np.abs(predicted["b1500"] - reference["b1500"]))),
        "b1500_mean_ssim": float(b1500_scores.mean()),
        "b1500_min_ssim": float(b1500_scores.min()),
    }
    return summary, {"adc_map": adc_maps, "b1500": b1500_maps}


def select_slice(reference: Dict[str, np.ndarray], slice_index: Optional[int]) -> int:
    max_index = reference["adc_map"].shape[0] - 1
    if slice_index is None:
        return reference["adc_map"].shape[0] // 2
    return max(0, min(slice_index, max_index))


def save_comparison_figure(
    h5_path: Path,
    output_path: Path,
    slice_index: int,
    reference: Dict[str, np.ndarray],
    comparisons: Sequence[Tuple[str, Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_rows = 2 * len(comparisons)
    fig, axes = plt.subplots(num_rows, 4, figsize=(18, 4 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = np.asarray([axes])
    elif num_rows > 1 and axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for block_idx, (label, predicted, summary, ssim_maps) in enumerate(comparisons):
        for metric_offset, metric_name in enumerate(METRIC_NAMES):
            row_idx = (2 * block_idx) + metric_offset
            ax_row = axes[row_idx]
            ref_slice = reference[metric_name][slice_index]
            pred_slice = predicted[metric_name][slice_index]
            diff_slice = np.abs(pred_slice - ref_slice)
            ssim_slice = ssim_maps[metric_name][slice_index]

            vmax = np.percentile(np.stack([ref_slice, pred_slice]), 99.5)
            ax_row[0].imshow(ref_slice, cmap="gray", vmin=0, vmax=vmax)
            ax_row[0].set_title(f"Stored {metric_name}")
            ax_row[1].imshow(pred_slice, cmap="gray", vmin=0, vmax=vmax)
            ax_row[1].set_title(f"{label} {metric_name}")
            ax_row[2].imshow(diff_slice, cmap="magma")
            ax_row[2].set_title(f"{metric_name} abs diff")
            ax_row[3].imshow(ssim_slice, cmap="viridis", vmin=0, vmax=1)
            mean_ssim = summary["adc_mean_ssim"] if metric_name == "adc_map" else summary["b1500_mean_ssim"]
            ax_row[3].set_title(f"{metric_name} SSIM map ({mean_ssim:.4f})")

            for ax in ax_row:
                ax.axis("off")

    fig.suptitle(f"{h5_path.name} | slice {slice_index}", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_rows(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "mode",
        "num_averages",
        "num_slices",
        "num_coils",
        "slice_index",
        "adc_mean_abs",
        "adc_max_abs",
        "adc_mean_ssim",
        "adc_min_ssim",
        "b1500_mean_abs",
        "b1500_max_abs",
        "b1500_mean_ssim",
        "b1500_min_ssim",
        "figure_path",
        "status",
        "error",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_h5_file(
    h5_path: Path,
    input_root: Optional[Path],
    output_dir: Path,
    slice_index: Optional[int],
) -> List[Dict[str, Any]]:
    logging.info("Processing %s", h5_path)
    payload = load_public_dwi_h5(h5_path)
    reference = payload["reference"]
    resolved_slice = select_slice(reference, slice_index)
    output_stem = safe_output_stem(h5_path, input_root)
    figure_path = output_dir / "figures" / f"{output_stem}.png"

    successful: List[Tuple[str, Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]] = []
    rows: List[Dict[str, Any]] = []

    base_row = {
        "file_path": str(h5_path),
        "num_averages": payload["kspace"].shape[0],
        "num_slices": payload["kspace"].shape[1],
        "num_coils": payload["kspace"].shape[2],
        "slice_index": resolved_slice,
        "figure_path": str(figure_path),
    }

    public_row = dict(base_row)
    public_row["mode"] = "public_csm"
    logging.info("  Mode public_csm")
    try:
        public_predicted = reconstruct_public_csm(
            kspace=payload["kspace"],
            calibration=payload["calibration"],
            coil_sens_maps=payload["coil_sens_maps"],
            hdr=payload["hdr"],
        )
        public_summary, public_ssim_maps = summarize_metrics(reference, public_predicted)
        public_row.update(public_summary)
        public_row["status"] = "ok"
        public_row["error"] = ""
        successful.append((MODE_DISPLAY_NAMES["public_csm"], public_predicted, public_summary, public_ssim_maps))
    except Exception as exc:  # noqa: PERF203
        logging.exception("  Mode public_csm failed for %s", h5_path)
        public_row["status"] = "error"
        public_row["error"] = repr(exc)
    rows.append(public_row)

    td_modes = ("td_esc_csm", "td_esc_espirit")
    td_rows = {mode_name: dict(base_row, mode=mode_name) for mode_name in td_modes}
    logging.info("  Modes td_esc_csm + td_esc_espirit")
    try:
        td_predictions = reconstruct_td_esc_variants(
            kspace=payload["kspace"],
            calibration=payload["calibration"],
            coil_sens_maps=payload["coil_sens_maps"],
            hdr=payload["hdr"],
        )
        for mode_name in td_modes:
            predicted = td_predictions[mode_name]
            summary, ssim_maps = summarize_metrics(reference, predicted)
            td_rows[mode_name].update(summary)
            td_rows[mode_name]["status"] = "ok"
            td_rows[mode_name]["error"] = ""
            successful.append((MODE_DISPLAY_NAMES[mode_name], predicted, summary, ssim_maps))
    except Exception as exc:  # noqa: PERF203
        logging.exception("  td/esc modes failed for %s", h5_path)
        for mode_name in td_modes:
            td_rows[mode_name]["status"] = "error"
            td_rows[mode_name]["error"] = repr(exc)

    rows.extend(td_rows[mode_name] for mode_name in td_modes)

    if successful:
        save_comparison_figure(
            h5_path=h5_path,
            output_path=figure_path,
            slice_index=resolved_slice,
            reference=reference,
            comparisons=successful,
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory searched recursively for input H5 files.")
    parser.add_argument("--input-h5s", nargs="*", type=Path, default=None, help="Optional explicit list of H5 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where figures and shard summaries will be written.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice index to plot. Defaults to the middle slice.")
    parser.add_argument("--job-index", type=int, default=0, help="Shard index for array jobs.")
    parser.add_argument("--job-count", type=int, default=1, help="Number of shards for array jobs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.input_dir is None and not args.input_h5s:
        raise ValueError("Provide --input-dir and/or --input-h5s.")

    all_h5s = discover_h5_files(args.input_dir, args.input_h5s)
    if not all_h5s:
        raise FileNotFoundError("No input H5 files found.")

    shard = shard_items(all_h5s, args.job_index, args.job_count)
    logging.info("Discovered %d H5 files; shard %d/%d will process %d", len(all_h5s), args.job_index, args.job_count, len(shard))
    if not shard:
        return

    summary_rows: List[Dict[str, Any]] = []
    for h5_path in shard:
        summary_rows.extend(
            process_h5_file(
                h5_path=h5_path,
                input_root=args.input_dir,
                output_dir=args.output_dir,
                slice_index=args.slice_index,
            )
        )

    summary_path = (
        args.output_dir
        / "summaries"
        / f"summary_job_{args.job_index:04d}_of_{args.job_count:04d}.csv"
    )
    write_summary_rows(summary_path, summary_rows)
    logging.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
