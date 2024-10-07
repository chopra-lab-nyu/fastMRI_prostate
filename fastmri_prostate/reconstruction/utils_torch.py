import torch
import numpy as np

from torch.fft import fftshift, ifftshift, ifftn
from typing import List, Optional, Sequence, Tuple


def ifftnd(kspace: torch.Tensor, axes: Optional[Sequence[int]] = [-1]) -> torch.Tensor:
    """
    Compute the n-dimensional inverse Fourier transform of the k-space data along the specified axes.

    Parameters:
    -----------
    kspace: torch.Tensor
        The input k-space data.
    axes: list or tuple, optional
        The list of axes along which to compute the inverse Fourier transform. Default is [-1].

    Returns:
    --------
    img: torch.Tensor
        The output image after inverse Fourier transform.
    """

    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, dim=axes), dim=axes), dim=axes)   
    img *= torch.sqrt(torch.prod(torch.tensor([kspace.shape[axis] for axis in axes], dtype=torch.float32)))    

    return img


def flip_im(vol: torch.Tensor, slice_axis: int) -> torch.Tensor:
    """
    Flips a 3D image volume along the slice axis.

    Parameters
    ----------
    vol : torch.Tensor of shape (slices, height, width)
        The 3D image volume to be flipped.
    slice_axis : int
        The slice axis along which to perform the flip

    Returns
    -------
    torch.Tensor
        The flipped 3D image volume 
    """

    for i in range(vol.shape[slice_axis]):
        vol[i] = torch.flip(vol[i], dims=[0])
    return vol

  
def center_crop_im(im_3d: torch.Tensor, crop_to_size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crop an image to a given size.
    
    Parameters:
    -----------
    im_3d : torch.Tensor
        Input image of shape (slices, x, y).
    crop_to_size : list
        List containing the target size for x and y dimensions.
    
    Returns:
    --------
    torch.Tensor
        Center cropped image of size {slices, x_cropped, y_cropped}. 
    """
    x_crop = im_3d.shape[-1] // 2 - crop_to_size[0] // 2
    y_crop = im_3d.shape[-2] // 2 - crop_to_size[1] // 2

    return im_3d[:, y_crop:y_crop + crop_to_size[1], x_crop:x_crop + crop_to_size[0]]  