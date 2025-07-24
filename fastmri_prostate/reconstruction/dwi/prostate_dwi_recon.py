import logging
import numpy as np
from pathlib import Path
from time import time
from typing import Dict, Tuple
import xml.etree.ElementTree as etree

from fastmri_prostate.reconstruction.dwi.regridding import trapezoidal_regridding
from fastmri_prostate.reconstruction.dwi.diffusion_metrics import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.grappa import Grappa
from fastmri_prostate.reconstruction.utils import ifftnd, flip_im, center_crop_im


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

    offset = 2 if img_vol.shape[0] == 48 else 0

    return {
        'b50x': np.sum(img_vol[2 - offset:21:6, ...][:num_b50_averages], axis=0) / num_b50_averages,
        'b50y': np.sum(img_vol[3 - offset:22:6, ...][:num_b50_averages], axis=0) / num_b50_averages,
        'b50z': np.sum(img_vol[4 - offset:23:6, ...][:num_b50_averages], axis=0) / num_b50_averages,
        'b1000x': np.sum(
            np.concatenate([
                img_vol[5 - offset:24:6, ...],
                img_vol[26 - offset:48:3, ...]
            ], axis=0)[:num_b1000_averages], axis=0
        ) / num_b1000_averages,
        'b1000y': np.sum(
            np.concatenate([
                img_vol[6 - offset:25:6, ...],
                img_vol[27 - offset:49:3, ...]
            ], axis=0)[:num_b1000_averages], axis=0
        ) / num_b1000_averages,
        'b1000z': np.sum(
            np.concatenate([
                img_vol[7 - offset:26:6, ...],
                img_vol[28 - offset:50:3, ...]
            ], axis=0)[:num_b1000_averages], axis=0
        ) / num_b1000_averages,
    }


def dwi_reconstruction(kspace: np.ndarray, calibration: np.ndarray, coil_sens_maps: np.ndarray, hdr: Dict) -> Dict:
    """ The reconstruction uses trapezoidal regridding to regrid the k-space data and computes GRAPPA weights for each slice 
    of the input k-space data using the calibration data. It applies the computed GRAPPA weights to the k-space data 
    to obtain image data, which is then combined with the coil sensitivity maps to reconstruct the DWI images. 
    The resulting images are cropped and returned as a dictionary with b50, b1000, trace, ADC, and b1500 values.

    Parameters:
    -----------
    kspace : np.ndarray
        The k-space data with dimensions (averages, slices, coils, readout, phase).
    calibration : np.ndarray
        The calibration data with dimensions (slices, coils, readout, phase).
    coil_sens_maps : np.ndarray
        The coil sensitivity maps with dimensions (slices, coils, readout, phase).
    hdr : dict
        The header information for the diffusion-weighted imaging.

    Returns:
    --------
    img_dict : dict
        A dictionary containing the reconstructed DW images and trace, ADC, and b1500 values

    """   
    
    kspace_slice_regridded = trapezoidal_regridding(kspace[0, 0, ...], hdr)
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    grappa_weight_dict = {}
    for slice_num in range(kspace.shape[1]):
        calibration_regridded = trapezoidal_regridding(calibration[slice_num, ...], hdr)
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0 ,1))
        )
    img_post_grappa = np.zeros(shape=kspace.shape, dtype=complex)
    
    for average in range(kspace.shape[0]):
        for slice_num in range(kspace.shape[1]):
            kspace_slice_regridded = trapezoidal_regridding(kspace[average, slice_num, ...], hdr)
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num]
            )
            img = ifftnd(kspace_post_grappa, [0, -1])
            img_post_grappa[average][slice_num] = np.transpose(img, (1, 2, 0))
        
        if average % 5 == 0:
            logging.info("Processed {0} averages of {1}".format(average, kspace.shape[0]))

    img_vol = np.zeros(shape=(kspace.shape[0], kspace.shape[1], kspace.shape[3], kspace.shape[4]), dtype=complex)

    for average in range(img_post_grappa.shape[0]):
        coil_comb_img = np.sum(img_post_grappa[average] * (coil_sens_maps.conj()), axis=1)
        img_vol[average] = coil_comb_img

    img_vol = np.abs(img_vol)

    img_dict = compute_averages(img_vol)
    img_dict = compute_trace_adc_b1500(img_dict)

    center_crop_size = (100, 100)
    for src_img in img_dict.keys():
        img_dict[src_img] = center_crop_im(flip_im(img_dict[src_img], 0), center_crop_size)

    return img_dict

            
