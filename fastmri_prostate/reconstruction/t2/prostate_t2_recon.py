import os
import numpy as np
from typing import Dict

from fastmri_prostate.data.mri_data import zero_pad_kspace_hdr
from fastmri_prostate.reconstruction.utils import center_crop_im, ifftnd
from fastmri_prostate.reconstruction.grappa import Grappa


def image_recon(kspace_post_grappa_all: np.ndarray, calib_data: np.ndarray, hdr: Dict, averages_to_use: int) -> Dict:
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_post_grappa_all.shape
    im_list = []
    for average in range(averages_to_use): 
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(kspace_grappa, hdr)
        coil_combined_image = create_coil_combined_im(kspace_grappa_padded)
        im_list.append(coil_combined_image)
    
    im = np.array(im_list)
    im_3d = np.mean(im, axis = 0) 
    # center crop image to 320 x 320
    img_dict = {}
    img_dict['reconstruction_rss'] = center_crop_im(im_3d, [320, 320]) 
    img_dict['kspace_post_grappa'] = kspace_post_grappa_all
    img_dict['calibration_data'] = calib_data

    return img_dict

def get_avg_to_pattern(kspace, num_avg):
    _, num_slices, _, _, _ = kspace.shape
    pe_line = kspace[num_avg, num_slices // 2, 0, 0, :]
    even_sum = pe_line[::2].sum()
    odd_sum  = pe_line[1::2].sum()
    if even_sum == 0:
        return 1
    elif odd_sum == 0:
        return 0
    else:
        raise Exception("Kspace does not follow any pattern")   

def t2_reconstruction(kspace_data: np.ndarray, calib_data: np.ndarray, hdr: Dict, averages_to_use: int) -> None:
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

    assert num_avg <= 3, "Number of averages must be less than oe equal to 3"
    
    avg_to_pattern = {i: get_avg_to_pattern(kspace_data, num_avg=i) for i in range(num_avg)}
    pattern_to_avg = {v: [k for k in avg_to_pattern if avg_to_pattern[k] == v] for v in set(avg_to_pattern.values())}

    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dicts = {k: {} for k in pattern_to_avg.keys()}

    grappa_objs = {}

    for k, v in pattern_to_avg.items():
        kspace_slice_regridded = kspace_data[v[0], 0, ...]
        grappa_objs[k] = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)
    
    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        for k, v in grappa_weight_dicts.items():
            v[slice_num] = grappa_objs[k].compute_weights(
                np.transpose(calibration_regridded, (2, 0 ,1))
            )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)

    for average in range(num_avg):
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_objs[avg_to_pattern[average]].apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dicts[avg_to_pattern[average]][slice_num]
            )
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(np.moveaxis(kspace_post_grappa, 0, 1), 1, 2)

    return image_recon(kspace_post_grappa_all, calib_data, hdr, averages_to_use)


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
