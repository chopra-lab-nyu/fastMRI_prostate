import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import warnings

import torch
import scipy.optimize

from fastmri_prostate.reconstruction.dwi.regridding import trapezoidal_regridding
from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.grappa import Grappa
from fastmri_prostate.reconstruction.utils import flip_im, center_crop_im


@dataclass
class DWIESCResult:
    """Container for ESC-based diffusion reconstruction outputs."""

    images: Dict[str, np.ndarray]
    esc_images_per_average: np.ndarray
    post_grappa_coil_images: np.ndarray
    kspace_esc: np.ndarray
    kspace_by_direction: Dict[str, np.ndarray]
    direction_indices: Dict[str, np.ndarray]

    def get_direction_kspace(self, direction: str, max_averages: Optional[int] = None) -> np.ndarray:
        """Return post-GRAPPA ESC k-space for a diffusion direction limited to the requested averages."""

        if direction not in self.direction_indices:
            raise KeyError(f"Unknown diffusion direction '{direction}'.")

        indices = self.direction_indices[direction]
        if max_averages is not None:
            if max_averages <= 0:
                raise ValueError("max_averages must be positive when provided.")
            if len(indices) < max_averages:
                raise ValueError(
                    f"Requested {max_averages} averages for {direction}, but only {len(indices)} are available."
                )
            indices = indices[:max_averages]

        return self.kspace_esc[np.asarray(indices, dtype=int), ...]

def get_direction_indices(num_averages: int) -> Dict[str, np.ndarray]:
    """Return canonical diffusion direction indices (4 for b50*, 12 for b1000*)."""

    if num_averages not in (48, 50):
        warnings.warn(
            "Unexpected number of averages (%d) for diffusion dataset; canonical mapping assumes 48 or 50." % num_averages,
            RuntimeWarning,
            stacklevel=2,
        )

    offset = 2 if num_averages == 48 else 0

    base_indices = {
        'b50x': np.arange(2, 22, 6),
        'b50y': np.arange(3, 23, 6),
        'b50z': np.arange(4, 24, 6),
        'b1000x': np.concatenate([np.arange(5, 25, 6), np.arange(26, 48, 3)]),
        'b1000y': np.concatenate([np.arange(6, 26, 6), np.arange(27, 49, 3)]),
        'b1000z': np.concatenate([np.arange(7, 27, 6), np.arange(28, 50, 3)]),
    }

    direction_map: Dict[str, np.ndarray] = {}
    for key, indices in base_indices.items():
        adjusted = indices - offset
        adjusted = adjusted[(adjusted >= 0) & (adjusted < num_averages)]
        direction_map[key] = adjusted.astype(int)

    return direction_map


def compute_averages(img_vol: np.ndarray, num_b50_averages: int = 4, num_b1000_averages: int = 12) -> Dict:
    """
    Computes the average of the given image volume for different diffusion-weighted directions.

    Parameters:
    ----------
        img_vol : np.ndarray
            The input image volume containing diffusion-weighted images.
        num_b50_averages : int
            The number of b50 averages to use. Default is 4.
        num_b1000_averages : int
            The number of b1000 averages to use. Default is 12.

    Returns:
    -------
        dict: A dictionary containing the computed averages for different diffusion-weighted directions.

    Notes:
    -----
    There are typically 4 averages for each b50 diffusion direction and 12 averages for each b1000 direction.
    """

    assert img_vol.shape[0] == 50 or img_vol.shape[0] == 48, "Num averages in DWI volumes can only be 50 or 48"

    direction_indices = get_direction_indices(img_vol.shape[0])

    def average_direction(direction: str, limit: int) -> np.ndarray:
        indices = direction_indices[direction]
        if len(indices) < limit:
            raise ValueError(
                f"Requested {limit} averages for {direction}, but only {len(indices)} are available."
            )
        return np.mean(img_vol[indices[:limit], ...], axis=0)

    return {
        'b50x': average_direction('b50x', num_b50_averages),
        'b50y': average_direction('b50y', num_b50_averages),
        'b50z': average_direction('b50z', num_b50_averages),
        'b1000x': average_direction('b1000x', num_b1000_averages),
        'b1000y': average_direction('b1000y', num_b1000_averages),
        'b1000z': average_direction('b1000z', num_b1000_averages),
    }


def emulated_single_coil_slice(kspace_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emulate a single coil following the original ESC implementation."""

    if kspace_slice.ndim != 3:
        raise ValueError("Expected kspace_slice with dims (coils, readout, phase)")

    kspace_batch = kspace_slice[None, ...]
    coil_imgs = np.fft.ifftshift(
        np.fft.ifftn(
            np.fft.fftshift(kspace_batch, axes=(2, 3)),
            axes=(2, 3)
        ),
        axes=(2, 3)
    )

    recon_rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=1))

    mask = np.zeros_like(recon_rss, dtype=bool)
    height, width = mask.shape[1:]
    roi_h = min(200, height)
    roi_w = min(150, width)
    d1 = max((height - roi_h) // 2, 0)
    d2 = max((width - roi_w) // 2, 0)
    mask[:, d1:d1 + roi_h, d2:d2 + roi_w] = True

    A = coil_imgs.transpose(0, 2, 3, 1)[mask]
    b = recon_rss[mask]

    if A.size == 0 or np.linalg.norm(b) == 0:
        weights = np.ones(kspace_slice.shape[0], dtype=np.complex64)
        weights /= np.linalg.norm(weights) + 1e-12
        kspace_esc = np.sum(kspace_slice * weights[:, None, None], axis=0)
        recon_esc = np.sum(coil_imgs[0] * weights[:, None, None], axis=0)
        return kspace_esc.astype(np.complex64), recon_rss[0], np.abs(recon_esc)

    scale = 1e3 / (np.sqrt(np.linalg.norm(b)) + 1e-12)
    A_scaled = A * scale
    b_scaled = b * scale

    x_ls = np.linalg.lstsq(A_scaled, b_scaled, rcond=None)[0]
    x0 = np.concatenate((x_ls.real, x_ls.imag))

    A_torch = torch.from_numpy(A_scaled).to(torch.complex64)
    b_torch = torch.from_numpy(b_scaled).to(torch.float64)
    sqrt_b = torch.sqrt(torch.clamp(b_torch, min=0.0) + 1e-12)

    def objective(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        half = x_flat.size // 2
        real = torch.tensor(x_flat[:half], dtype=torch.float32, requires_grad=True)
        imag = torch.tensor(x_flat[half:], dtype=torch.float32, requires_grad=True)
        x_complex = torch.complex(real, imag)

        Ax = torch.matmul(A_torch, x_complex)
        loss = torch.sum((torch.sqrt(torch.abs(Ax) + 1e-12) - sqrt_b.float()) ** 2)
        loss.backward()

        grad = np.concatenate((real.grad.numpy(), imag.grad.numpy())).astype(np.float64)
        return loss.item(), grad

    x_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(objective, x0, iprint=-1)
    weights = x_opt[:x_opt.size // 2] + 1j * x_opt[x_opt.size // 2:]

    kspace_esc = np.sum(kspace_slice * weights[:, None, None], axis=0)
    recon_esc = np.sum(coil_imgs[0] * weights[:, None, None], axis=0)

    return kspace_esc.astype(np.complex64), recon_rss[0], np.abs(recon_esc)


def dwi_reconstruction_esc(
    kspace: np.ndarray,
    calibration: np.ndarray,
    hdr: Dict,
    num_b50_averages: int = 4,
    num_b1000_averages: int = 12,
    directions: Optional[Sequence[str]] = None,
    compute_metrics: bool = True,
) -> DWIESCResult:
    """Run GRAPPA + emulated single-coil (ESC) reconstruction.

    Parameters
    ----------
    kspace : np.ndarray
        Raw DWI k-space data with shape (averages, slices, coils, readout, phase).
    calibration : np.ndarray
        Fully-sampled calibration data with shape (slices, coils, readout, phase).
    hdr : Dict
        Acquisition header dictionary required for trapezoidal regridding.
    num_b50_averages : int, optional
        Number of b50 averages used when generating mean images.
    num_b1000_averages : int, optional
        Number of b1000 averages used when generating mean images.
    directions : Sequence[str], optional
        Subset of diffusion directions to keep in the outputs. Defaults to all available directions.
    compute_metrics : bool, optional
        Whether to compute trace/ADC/b1500 maps (requires all six diffusion directions).

    Returns
    -------
    DWIESCResult
        Dataclass bundle with ESC images, ESC k-space, post-GRAPPA coil-domain images,
        and direction-wise groupings.
    """

    kspace_slice_regridded = trapezoidal_regridding(kspace[0, 0, ...], hdr)
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    grappa_weight_dict = {}
    for slice_num in range(kspace.shape[1]):
        calibration_regridded = trapezoidal_regridding(calibration[slice_num, ...], hdr)
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0 ,1))
        )

    img_vol = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[3], kspace.shape[4]), dtype=float)
    kspace_esc_vol = np.zeros_like(img_vol, dtype=np.complex64)
    post_grappa_img_vol = np.zeros(
        (kspace.shape[0], kspace.shape[1], kspace.shape[2], kspace.shape[3], kspace.shape[4]),
        dtype=np.complex64,
    )

    for average in range(kspace.shape[0]):
        for slice_num in range(kspace.shape[1]):
            kspace_slice_regridded = trapezoidal_regridding(kspace[average, slice_num, ...], hdr)
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num]
            )

            # Bring coil dimension to leading position for ESC
            kspace_coil_first = np.transpose(kspace_post_grappa, (1, 2, 0))
            coil_domain = np.fft.ifftshift(
                np.fft.ifftn(
                    np.fft.fftshift(kspace_coil_first, axes=(1, 2)),
                    axes=(1, 2),
                ),
                axes=(1, 2),
            )
            esc_kspace, _, esc_image = emulated_single_coil_slice(kspace_coil_first)
            img_vol[average, slice_num] = esc_image
            kspace_esc_vol[average, slice_num] = esc_kspace
            post_grappa_img_vol[average, slice_num] = coil_domain

        if average % 5 == 0:
            logging.info("Processed {0} averages of {1}".format(average, kspace.shape[0]))

    direction_indices = get_direction_indices(kspace.shape[0])
    if directions is None:
        selected_directions = list(direction_indices.keys())
    else:
        selected_directions = list(directions)
        invalid = [d for d in selected_directions if d not in direction_indices]
        if invalid:
            raise ValueError(f"Unknown diffusion directions requested: {invalid}")

    kspace_by_direction: Dict[str, np.ndarray] = {}
    for direction in selected_directions:
        idx = direction_indices[direction]
        kspace_by_direction[direction] = kspace_esc_vol[idx, ...]

    img_dict_full = compute_averages(img_vol, num_b50_averages, num_b1000_averages)
    required_dirs = {'b50x', 'b50y', 'b50z', 'b1000x', 'b1000y', 'b1000z'}
    have_required = required_dirs.issubset(direction_indices.keys()) and required_dirs.issubset(set(selected_directions))

    if compute_metrics and have_required:
        img_dict_full = compute_trace_adc_b1500(img_dict_full)
    elif compute_metrics and not have_required:
        warnings.warn(
            "Skipping trace/ADC/b1500 computation because not all six diffusion directions were requested.",
            RuntimeWarning,
            stacklevel=2,
        )

    selected_set = set(selected_directions)
    img_dict = {
        key: value
        for key, value in img_dict_full.items()
        if (key in selected_set) or (key not in required_dirs)
    }

    center_crop_size = (100, 100)
    for src_img in img_dict.keys():
        img_dict[src_img] = center_crop_im(flip_im(img_dict[src_img], 0), center_crop_size)

    return DWIESCResult(
        images=img_dict,
        esc_images_per_average=img_vol,
        post_grappa_coil_images=post_grappa_img_vol,
        kspace_esc=kspace_esc_vol,
        kspace_by_direction=kspace_by_direction,
        direction_indices=direction_indices,
    )


def dwi_reconstruction(
    kspace: np.ndarray,
    calibration: np.ndarray,
    coil_sens_maps: np.ndarray,
    hdr: Dict,
    num_b50_averages: int = 4,
    num_b1000_averages: int = 12,
) -> Dict[str, np.ndarray]:
    """Backward-compatible wrapper that runs the ESC pipeline.

    Parameters
    ----------
    coil_sens_maps : np.ndarray
        Ignored. Retained for API compatibility with legacy callers.
    """

    warnings.warn(
        "`dwi_reconstruction` now runs the ESC workflow; `coil_sens_maps` is ignored. "
        "Use `dwi_reconstruction_esc` for structured outputs.",
        DeprecationWarning,
        stacklevel=2,
    )

    result = dwi_reconstruction_esc(
        kspace,
        calibration,
        hdr,
        num_b50_averages=num_b50_averages,
        num_b1000_averages=num_b1000_averages,
    )
    return result.images


def dwi_reconstruction_esc_kspace(
    kspace: np.ndarray,
    calibration: np.ndarray,
    hdr: Dict,
    directions: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Convenience wrapper returning only ESC k-space grouped by diffusion direction."""

    result = dwi_reconstruction_esc(
        kspace,
        calibration,
        hdr,
        directions=directions,
        compute_metrics=False,
    )
    return result.kspace_by_direction
