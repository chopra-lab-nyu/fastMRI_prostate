import logging
import numpy as np
from typing import Optional


# FFT/IFFT helpers with ortho normalization
_fft = lambda x, ax: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)
_ifft = lambda X, ax: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax)


def espirit(X: np.ndarray, k: int, r: int, t: float, c: float) -> np.ndarray:
    """
    Derives the ESPIRiT operator.

    Arguments:
        X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric 
           dimensions and (nc) is the channel dimension.
        k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel 
           will have dimensions (1, k, k, 8)
        r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the 
           calibration region will have dimensions (1, r, r, 8)
        t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
           largest singular value are set to zero.
        c: Crop threshold that determines eigenvalues "=1".
        
    Returns:
        maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
              being the idx'th set of ESPIRiT maps.
    """

    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]
    
    logging.debug("ESPIRiT: input shape (%d, %d, %d, %d), kernel=%d, calib=%d", sx, sy, sz, nc, k, r)

    sxt = (sx // 2 - r // 2, sx // 2 + r // 2) if (sx > 1) else (0, 1)
    syt = (sy // 2 - r // 2, sy // 2 + r // 2) if (sy > 1) else (0, 1)
    szt = (sz // 2 - r // 2, sz // 2 + r // 2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)
    logging.debug("ESPIRiT: calibration region shape %s", C.shape)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r - k + 1) ** p, k ** p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            for zdx in range(max(1, C.shape[2] - k + 1)):
                # numpy handles when the indices are too big
                block = C[xdx:xdx + k, ydx:ydx + k, zdx:zdx + k, :].astype(np.complex64)
                A[idx, :] = block.flatten()
                idx = idx + 1
    
    logging.debug("ESPIRiT: Hankel matrix shape %s, computing SVD...", A.shape)

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]
    logging.debug("ESPIRiT: selected %d kernels from SVD", n)

    kxt = (sx // 2 - k // 2, sx // 2 + k // 2) if (sx > 1) else (0, 1)
    kyt = (sy // 2 - k // 2, sy // 2 + k // 2) if (sy > 1) else (0, 1)
    kzt = (sz // 2 - k // 2, sz // 2 + k // 2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1], kyt[0]:kyt[1], kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    # Take the iucfft
    logging.debug("ESPIRiT: computing kernel images...")
    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            kerimgs[:, :, :, jdx, idx] = _fft(ker, axes) * np.sqrt(sx * sy * sz) / np.sqrt(k ** p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    logging.debug("ESPIRiT: computing point-wise eigenvalue decomposition (%d x %d x %d pixels)...", sx, sy, sz)
    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    total_pixels = sx * sy * sz
    pixel_count = 0
    log_interval = max(1, total_pixels // 10)  # Log every 10%
    
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):
                Gq = kerimgs[idx, jdx, kdx, :, :]
                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if s[ldx] ** 2 > c:
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]
                
                pixel_count += 1
                if pixel_count % log_interval == 0:
                    logging.debug("ESPIRiT: eigenvalue decomposition %d%% complete", int(100 * pixel_count / total_pixels))
    
    logging.debug("ESPIRiT: completed, output maps shape %s", maps.shape)
    return maps


def espirit_maps_from_calib(
    calib_kspace: np.ndarray,
    calib_width: int = 32,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop_thresh: float = 0.9,
) -> np.ndarray:
    """
    Estimate coil sensitivity maps using ESPIRiT.

    Parameters
    ----------
    calib_kspace : np.ndarray
        Calibration k-space with shape (coils, x, y).
    calib_width : int
        Width/height of the calibration region for ESPIRiT.
    thresh : float
        Singular value threshold (relative to max) for selecting kernels.
    kernel_width : int
        Kernel width for calibration.
    crop_thresh : float
        Eigenvalue threshold for determining "valid" sensitivity values.

    Returns
    -------
    np.ndarray
        Sensitivity maps with shape (coils, x, y). Returns the first set of maps.
    """
    # Input: (coils, x, y) -> need (1, x, y, coils) for espirit function
    coils, nx, ny = calib_kspace.shape
    
    # Ensure calib_width doesn't exceed data dimensions
    calib_width = min(calib_width, nx, ny)
    
    kspace_4d = calib_kspace.transpose(1, 2, 0)[np.newaxis, ...]  # (1, x, y, coils)

    # Run ESPIRiT - returns (1, x, y, coils, coils)
    esp_maps = espirit(kspace_4d, kernel_width, calib_width, thresh, crop_thresh)

    # Extract first set of maps: (1, x, y, coils, 0) -> (x, y, coils) -> (coils, x, y)
    maps_first_set = esp_maps[0, :, :, :, 0]  # (x, y, coils)
    maps_coils_first = maps_first_set.transpose(2, 0, 1)  # (coils, x, y)

    return maps_coils_first


def combine_with_maps(coil_images: np.ndarray, sens_maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Combination given coil images and sensitivity maps.

    Parameters
    ----------
    coil_images : np.ndarray
        Complex coil images with shape (coils, x, y).
    sens_maps : np.ndarray
        Complex sensitivity maps with shape (coils, x, y).
    eps : float
        Unused; retained for signature compatibility.

    Returns
    -------
    np.ndarray
        Combined magnitude image with shape (x, y).
    """

    del eps
    return np.abs(np.sum(np.conj(sens_maps) * coil_images, axis=0))
