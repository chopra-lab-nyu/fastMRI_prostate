import numpy as np
from typing import Dict, Iterable, Sequence, Tuple

from fastmri_prostate.data.mri_data import zero_pad_kspace_hdr
from fastmri_prostate.reconstruction.utils import center_crop_im, ifftnd
from fastmri_prostate.reconstruction.grappa import Grappa


DEFAULT_T2_AVERAGING_SCHEMES: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("axt2_1", (1,)),
    ("axt2_2", (2,)),
    ("axt2_1_2", (1, 2)),
    ("axt2_2_3", (2, 3)),
    ("axt2_1_3", (1, 3)),
    ("axt2_1_2_3", (1, 2, 3)),
)


def _canonicalize_averages(averages: Iterable[int]) -> Tuple[int, ...]:
    ordered = sorted({int(avg) for avg in averages})
    if not ordered:
        raise ValueError("Average scheme cannot be empty.")
    return tuple(ordered)


def _normalize_averaging_schemes(
    num_avg: int,
    averaging_schemes: Sequence[Tuple[str, Sequence[int]]] | None,
) -> list[Tuple[str, Tuple[int, ...]]]:
    if averaging_schemes is None:
        default = tuple(range(1, num_avg + 1))
        return [(f"axt2_{'_'.join(str(idx) for idx in default)}", default)]

    normalized: list[Tuple[str, Tuple[int, ...]]] = []
    for raw_tag, raw_indices in averaging_schemes:
        scheme = _canonicalize_averages(raw_indices)
        tag = raw_tag.strip() if raw_tag else f"axt2_{'_'.join(str(idx) for idx in scheme)}"
        normalized.append((tag, scheme))

    return normalized


def image_recon(
    kspace_post_grappa_all: np.ndarray,
    calib_data: np.ndarray,
    hdr,
    averaging_schemes: Sequence[Tuple[str, Sequence[int]]] | None = None,
    store_kspace: bool = True,
) -> Dict:
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_post_grappa_all.shape
    im_list = []
    for average in range(num_avg):
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(kspace_grappa, hdr)
        coil_combined_image = create_coil_combined_im(kspace_grappa_padded)
        im_list.append(coil_combined_image)

    im = np.asarray(im_list)
    averaging_schemes = _normalize_averaging_schemes(num_avg, averaging_schemes)

    per_average = np.asarray([center_crop_im(im[avg], [320, 320]) for avg in range(num_avg)], dtype=np.float32)
    reconstruction_rss = center_crop_im(np.mean(im, axis=0), [320, 320]).astype(np.float32)

    img_dict = {}
    img_dict["reconstruction_rss"] = reconstruction_rss
    img_dict["images/per_average"] = per_average
    img_dict["metadata/averaging_schemes"] = np.asarray([tag for tag, _ in averaging_schemes], dtype="S24")

    for tag, one_based_indices in averaging_schemes:
        zero_based_indices = np.asarray(one_based_indices, dtype=int) - 1
        averaged = np.mean(im[zero_based_indices, ...], axis=0)
        img_dict[f"images/averaged/{tag}"] = center_crop_im(averaged, [320, 320]).astype(np.float32)
        img_dict[f"metadata/average_indices/{tag}"] = np.asarray(one_based_indices, dtype=np.int16)

    if store_kspace:
        img_dict["kspace_post_grappa"] = kspace_post_grappa_all
        img_dict["kspace/post_grappa_full"] = kspace_post_grappa_all

    return img_dict


def t2_reconstruction(
    kspace_data: np.ndarray,
    calib_data: np.ndarray,
    hdr: Dict,
    averaging_schemes: Sequence[Tuple[str, Sequence[int]]] | None = None,
    store_kspace: bool = True,
) -> Dict:
    """
    Perform T2-weighted image reconstruction using GRAPPA technique.

    Parameters:
    -----------
    kspace_data: numpy.ndarray
        Input k-space data with shape (num_aves, num_slices, num_coils, num_ro, num_pe)
    calib_data: numpy.ndarray
        Calibration data for GRAPPA with shape (num_slices, num_coils, num_pe_cal)
    hdr:
        Dict
         
    Returns:
    --------
    im_final: numpy.ndarray
        Reconstructed image with shape (num_slices, 320, 320)
    """
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_data.shape
    if num_avg < 1:
        raise ValueError(f"T2 reconstruction requires at least 1 average, found {num_avg}.")

    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}

    kspace_slice_regridded = kspace_data[0, 0, ...]
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)
    if num_avg > 1:
        grappa_weight_dict_2 = {}
        kspace_slice_regridded_2 = kspace_data[1, 0, ...]
        grappa_obj_2 = Grappa(np.transpose(kspace_slice_regridded_2, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)
    else:
        grappa_weight_dict_2 = grappa_weight_dict
        grappa_obj_2 = grappa_obj
    
    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0 ,1))
        )
        if num_avg > 1:
            grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
                np.transpose(calibration_regridded, (2, 0 ,1))
            )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)

    for average in range(num_avg):
        if average % 2 == 0:
            grappa_obj_cur = grappa_obj
            grappa_weights_cur = grappa_weight_dict
        else:
            grappa_obj_cur = grappa_obj_2
            grappa_weights_cur = grappa_weight_dict_2
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_obj_cur.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weights_cur[slice_num]
            )
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(np.moveaxis(kspace_post_grappa, 0, 1), 1, 2)

    return image_recon(
        kspace_post_grappa_all,
        calib_data,
        hdr,
        averaging_schemes=averaging_schemes,
        store_kspace=store_kspace,
    )


def create_coil_combined_im(multicoil_multislice_kspace: np.ndarray) -> np.ndarray:
    """
    Create a coil combined image from a multicoil-multislice k-space array.
    
    Parameters:
    -----------
    multicoil_multislice_kspace : array-like
        Input k-space data with shape (slices, coils, readout, phase encode).
    
    Returns:
    --------
    image_mat : array-like
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = np.zeros((k.shape[0], k.shape[2], k.shape[3]))     
    for i in range(image_mat.shape[0]):                             
        data_sl = k[i,:,:,:]                                        
        image = ifftnd(data_sl, [1,2])                             
        image = rss(image, axis = 0)                         
        image_mat[i,:,:] = np.flipud(image)                         
    return image_mat
    

def rss(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the Root Sum-of-Squares (RSS) value of a complex signal along a specified axis.

    Parameters
    ----------
    sig : np.ndarray
        The complex signal to compute the RMS value of.
    axis : int, optional
        The axis along which to compute the RMS value. Default is -1.

    Returns
    -------
    rss : np.ndarray
        The RSS value of the complex signal along the specified axis.
    """
    return np.sqrt(np.sum(abs(sig)**2, axis))
