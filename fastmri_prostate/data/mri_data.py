import json
import logging
import struct
from pathlib import Path
import h5py
import numpy as np
import xml.etree.ElementTree as etree
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def get_slice_order(hdr):
    temp = hdr['Config']['chronSliceIndices']
    temp = ' '.join(temp.split())
    temp = [int(i) for i in temp.split()]
    slice_order = [i for i in temp if i != -1] 
    
    slice_order = np.concatenate([np.where(slice_order == i)[0] for i in np.arange(len(slice_order))])

    return slice_order


def _require_twixtools():
    try:
        import twixtools  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "twixtools is required to load Siemens .dat files. Install via `pip install twixtools`."
        ) from exc
    return twixtools


def load_dat_file_T2(raw_dat_file: str) -> Tuple: 
    """
    Load T2 fastmri file.
    
    Parameters:
    -----------
    fname : str
        Path to the h5 fastmri file.
    
    Returns:

    """
    import twixtools
    twix = twixtools.read_twix(str(raw_dat_file))
    mapped = twixtools.map_twix(twix)

    im_data = mapped[-1]['image']
    refscan_data = np.squeeze(mapped[-1]['refscan'][:])
    hdr = mapped[-1]['hdr']

    im_data.flags['remove_os'] = False
    im_data.flags['average']['Ave'] = False

    data = im_data[:].squeeze()

    slice_order = get_slice_order(hdr)
    data = data[slice_order, ...]
    refscan_data = refscan_data[slice_order, ...]

    data = np.transpose(data, (1, 0, 3, 4, 2))
    refscan_data = np.transpose(refscan_data, (0, 2, 3, 1))

    data = np.flip(data, axis = 1)
    refscan_data = np.flip(refscan_data, axis = 0)

    return data, refscan_data, hdr
    

def _zero_pad_along_axis(arr: np.ndarray, axis: int, target_size: int) -> np.ndarray:
    """Zero-pad an array along the specified axis to reach the target size."""

    current = arr.shape[axis]
    if current >= target_size:
        return arr

    pad_total = target_size - current
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(arr, pad_width, mode='constant')


def _extract_epi_params(hdr: Dict) -> Dict[str, float]:
    """Extract the subset of header parameters required for regridding."""

    config = hdr.get('Config', {})
    return {
        'regridrampuptime': float(config.get('RampupTime', 0.0)),
        'regridrampdowntime': float(config.get('RampdownTime', 0.0)),
        'regridflattoptime': float(config.get('FlattopTime', 0.0)),
        'regriddelaytime': float(config.get('DelaySamplesTime', 0.0)),
        'regridadcduration': float(config.get('ADCDuration', 0.0)),
        'regriddestsamples': float(config.get('DestSamples', 0.0)),
        'echospacing': float(config.get('RampUpTime', 0.0))
        + float(config.get('RampDownTime', 0.0))
        + float(config.get('FlatTopTime', 0.0)),
    }


def load_dat_file_dwi(raw_dat_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Siemens diffusion `.dat` file and return k-space, calibration, and metadata.

    The metadata dictionary contains the trapezoidal regridding parameters and the
    patient identifier extracted from the TWIX header (``metadata['patient_id']``).
    """

    import twixtools
    twix = twixtools.read_twix(str(raw_dat_file))
    mapped = twixtools.map_twix(twix)

    im_data = mapped[-1]['image']
    refscan = mapped[-1]['refscan']
    hdr = mapped[-1]['hdr']

    im_data.flags['remove_os'] = False
    im_data.flags['average']['Ave'] = False

    kspace = im_data[:].squeeze()
    calibration = refscan[:].squeeze()

    slice_order = get_slice_order(hdr)
    kspace = kspace[:, slice_order, ...]
    calibration = calibration[slice_order, ...]
    
    if kspace.shape[2] > calibration.shape[1]:
        calibration = _zero_pad_along_axis(calibration, axis=1, target_size=kspace.shape[2])

    # Reorder to match reconstruction expectations
    kspace = np.transpose(kspace, (0, 1, 3, 4, 2)).copy()
    calibration = np.transpose(calibration, (0, 2, 3, 1)).copy()

    epi_params = _extract_epi_params(hdr)
    patient_id = hdr.get('Config', {}).get('PatientID') if isinstance(hdr, dict) else None
    if isinstance(patient_id, bytes):
        patient_id = patient_id.decode(errors='ignore')

    regrid_params = {
        'rampUpTime': epi_params['regridrampuptime'],
        'rampDownTime': epi_params['regridrampdowntime'],
        'flatTopTime': epi_params['regridflattoptime'],
        'acqDelayTime': epi_params['regriddelaytime'],
        'adcDuration': epi_params['regridadcduration'],
        'destSamples': epi_params['regriddestsamples'],
        'echoSpacing': epi_params['echospacing'],
        'patient_id': patient_id,
    }

    return kspace, calibration, regrid_params


def load_file_T2(fname: str) -> Tuple:
    """
    Load T2 fastmri file.
    
    Parameters:
    -----------
    fname : str
        Path to the h5 fastmri file.
    
    Returns:
    --------
    Tuple
        A tuple containing the kspace, calibration_data, hdr, im_recon, and attributes of the file.
    """

    with h5py.File(fname, "r") as hf:
        kspace = hf["kspace"][:]       
        calibration_data = hf["calibration_data"][:] 
        hdr = hf["ismrmrd_header"][()]
        im_recon = hf["reconstruction_rss"][:]   
        atts = dict()
        atts['max'] = hf.attrs['max']
        atts['norm'] = hf.attrs['norm']
        atts['patient_id'] = hf.attrs['patient_id']
        atts['acquisition'] = hf.attrs['acquisition']

    return kspace, calibration_data, hdr, im_recon, atts


def load_file_dwi(fname: str) -> Tuple:
    """
    Load DWI fastmri file.
    
    Parameters:
    -----------
    fname : str
        Path to the h5 fastmri file.
    
    Returns:
    --------
    Tuple
        A tuple containing the kspace, calibration_data, hdr, and coil sensitivity maps.
    """

    with h5py.File(fname, 'r') as f:
        kspace = f['kspace'][:]
        calibration = f['calibration_data'][:]
        coil_sens_maps = f['coil_sens_maps'][:]
        #phase_corr = f['phase_correction'][:]
        
        ismrmrd_header = f['ismrmrd_header'][()]
        hdr = get_regridding_params(ismrmrd_header)
    
    return kspace, calibration, coil_sens_maps, hdr


def get_padding_from_xml(hdr: str) -> float:
    """
    Extract the padding value from an XML header string.

    Parameters:
    -----------
    hdr : str
        The XML header string.

    Returns:
    --------
    float
        The padding value calculated as (x - max_enc)/2, where x is the readout dimension and 
        max_enc is the maximum phase-encoding dimension.
    """
    et_root = etree.fromstring(hdr)                                                              
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]                              
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1                              
    enc = ["encoding", "encodedSpace", "matrixSize"]                                              
    enc_x = int(et_query(et_root, enc + ["x"]))                                                  
    padding = (enc_x - enc_limits_max)/2                                                         

    return padding


def get_padding_from_hdr(hdr):
    enc_size_1 = int(hdr['Config']['NPeFTLen'])  # Readout (x) dimension
    enc_limits_max = int(hdr['Config']['NLinMeas'])  # Max phase-encoding dimension

    padding = (enc_size_1 - enc_limits_max) / 2.0
    return padding


def get_padding(data_shape: Tuple) -> int:
    try:
        #max_enc = int(hdr['MeasYaps']['sKSpace']['lPhaseEncodingLines']) + 1
        #enc_x = int(hdr['MeasYaps']['sKSpace']['lBaseResolution'])
        ro, pe = data_shape[-2], data_shape[-1]
        padding = (ro - pe) / 2

        return padding

    except KeyError as e:
        print(f"Key error: {e}")
        return None


def et_query(root: etree.Element, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD") -> str:
    """
    ElementTree query function.
    
    This function queries an XML document using ElementTree.
    
    Parameters:
    -----------
    root : Element
        Root of the XML document to search through.
    qlist : Sequence of str
        A sequence of strings for nested searches, e.g., ["Encoding", "matrixSize"].
    namespace : str, optional
        XML namespace to prepend query.
    
    Returns:
    --------
    str
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def zero_pad_kspace_hdr(unpadded_kspace: np.ndarray, hdr) -> np.ndarray:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    unpadded_kspace : array-like of shape (sl, ro , coils, pe)
        The k-space data to be padded.
    hdr : Dict

    Returns
    -------
    padded_kspace : ndarray of shape (sl, ro_padded, coils, pe_padded)
        The zero-padded k-space data, where ro_padded and pe_padded are
        the dimensions of the readout and phase-encoding directions after
        padding.

    Notes
    -----
    The padding value is calculated using the `get_padding` function, which
    extracts the padding value from either a dictionary or XML header string. 
    If the difference between the readout dimension and the maximum phase-encoding 
    dimension is not divisible by 2, the padding is applied asymmetrically, with one
    side having an additional zero-padding.
    """
    if isinstance(hdr, dict):
        padding = get_padding_from_hdr(hdr)
    else:
        padding = get_padding(unpadded_kspace.shape)                                                                    
    if padding%2 != 0:
        padding_left = int(np.floor(padding))                                                    
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(unpadded_kspace, ((0,0),(0,0),(0,0),(padding_left, padding_right)))     

    return padded_kspace


def get_regridding_params(hdr: str) -> Dict:
    """
    Extracts regridding parameters from header XML string.

    Parameters
    ----------
    hdr : str
        Header XML string.

    Returns
    -------
    dict
        A dictionary containing the extracted parameters.

    """
    res = {
        'rampUpTime': None,
        'rampDownTime': None,
        'flatTopTime': None,
        'acqDelayTime': None,
        'echoSpacing': None
    }
    
    et_root = etree.fromstring(hdr)
    namespace = {'ns': "http://www.ismrm.org/ISMRMRD"}

    for node in et_root.findall('ns:encoding/ns:trajectoryDescription/ns:userParameterLong', namespace):
        if node[0].text in res.keys():
            res[node[0].text] = float(node[1].text)
    
    return res


def save_recon(outp_dict: Dict[str, any], hdr: Dict, output_path: str) -> None:
    """
    Save reconstruction results to an HDF5 file.

    Parameters
    ----------
    outp_dict : dict
        A dictionary containing the reconstructed images, with the image names as keys.
    hdr : dict
        A dictionary containing the header information.
    output_path : str
        The file path to save the reconstructed images.

    Returns
    -------
    None
    """

    with h5py.File(output_path, "w") as hf:
        for key, outp in outp_dict.items():
            hf.create_dataset(key, data=outp)
        
        hdr_json = json.dumps(hdr)
        hf.create_dataset("hdr", data=hdr_json)

        ## To load and parse the JSON string to get the hdr dictionary
        # hdr_json = hf["hdr"][()]
        # hdr = json.loads(hdr_json)


def load_dwi_esc_h5(h5_path: Union[str, Path]) -> Dict[str, Any]:
    """Load an ESC DWI reconstruction HDF5 file into a nested dictionary.

    Parameters
    ----------
    h5_path : str or Path
        Path to the .h5 file produced by `fastmri_prostate_recon_from_dat`.

    Returns
    -------
    dict
        Flat dictionary keyed by HDF5 dataset paths, e.g.:
            - images/averaged/<direction>
            - images/per_average/<direction>
            - images/esc_full
            - kspace/esc/<direction>
            - kspace/post_grappa_full
            - metadata/directions
            - metadata/averages
            - metadata/direction_indices/<direction>
            - metrics/* (if present)
            - hdr (parsed JSON dict)
    """
    result: Dict[str, Any] = {}

    def _decode(val: Any) -> Any:
        if isinstance(val, bytes):
            return val.decode(errors="ignore")
        if isinstance(val, np.ndarray) and val.dtype.kind == "S":
            return [v.decode(errors="ignore") for v in val.flat]
        return val

    with h5py.File(h5_path, "r") as hf:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if name == "hdr":
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        data = _decode(data)
                elif isinstance(data, (bytes, np.ndarray)) and getattr(data, "dtype", None) is not None and data.dtype.kind == "S":
                    data = _decode(data)
                result[name] = data

        hf.visititems(visitor)

    return result
