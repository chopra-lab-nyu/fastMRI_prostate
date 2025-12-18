import numpy as np
from typing import Optional

import sigpy as sp
from sigpy.mri.app import EspiritCalib


def _ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT over the last two axes."""

    return np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.fftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1),
            norm=None,
        ),
        axes=(-2, -1),
    )


def espirit_maps_from_calib(
    calib_kspace: np.ndarray,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
) -> np.ndarray:
    """
    Estimate coil sensitivity maps using sigpy's ESPIRiT implementation.

    Parameters
    ----------
    calib_kspace : np.ndarray
        Calibration k-space with shape (coils, x, y).
    calib_width : int
        Width/height of the central calibration region.
    thresh : float
        Eigenvalue threshold for ESPIRiT.
    kernel_width : int
        Kernel width for calibration.

    Returns
    -------
    np.ndarray
        Sensitivity maps with shape (coils, x, y).
    """

    # Center-crop calibration
    cx, cy = calib_kspace.shape[-2:]
    sx = max((cx - calib_width) // 2, 0)
    sy = max((cy - calib_width) // 2, 0)
    ex = sx + min(calib_width, cx)
    ey = sy + min(calib_width, cy)
    calib_crop = calib_kspace[..., sx:ex, sy:ey]

    app = EspiritCalib(
        calib_crop,
        calib_width=calib_crop.shape[-1],
        thresh=thresh,
        kernel_width=kernel_width,
        max_iter=30,
        device=sp.Device(-1),  # CPU
        show_pbar=False,
    )
    maps = app.run()
    return np.asarray(maps)


def combine_with_maps(coil_images: np.ndarray, sens_maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    SENSE-style combination given coil images and sensitivity maps.

    Parameters
    ----------
    coil_images : np.ndarray
        Complex coil images with shape (coils, x, y).
    sens_maps : np.ndarray
        Complex sensitivity maps with shape (coils, x, y).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Combined magnitude image with shape (x, y).
    """

    num = np.sum(np.conj(sens_maps) * coil_images, axis=0)
    denom = np.sum(np.abs(sens_maps) ** 2, axis=0) + eps
    return np.abs(num / denom)
