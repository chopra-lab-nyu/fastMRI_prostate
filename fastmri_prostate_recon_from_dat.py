import argparse
import logging
from pathlib import Path

from fastmri_prostate.reconstruction.t2.prostate_t2_recon import t2_reconstruction
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import dwi_reconstruction
from fastmri_prostate.data.mri_data import load_dat_file_T2, load_file_dwi, save_recon

log_dir = Path('recon_logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / 'recon.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(args):
    Path(args['output_path']).mkdir(exist_ok=True)

    t2_dat_files = list(Path(args['data_path']).glob('*AXT2*.dat'))

    t2_file_paths = []
    for file in t2_dat_files: 
        t2_file_paths.append({
            'input_file': file,
            'output_file': f"{Path(args['output_path']) / file.stem}.h5"
        })

    t2_file_dict = t2_file_paths[args['index']]
    if args['sequence'] == 't2':
        kspace, calibration_data, hdr = load_dat_file_T2(t2_file_dict['input_file'])
        if kspace is None:
            logging.warning(f"File {t2_file_dict['input_file']} does not have kspace data, skipping...")
        else:
            img_dict = t2_reconstruction(kspace, calibration_data, hdr)
            save_recon(img_dict, hdr, t2_file_dict['output_file'])

            logging.info(f"saved {t2_file_dict['input_file']} to {t2_file_dict['output_file']}")
    
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prostate T2/DWI reconstruction')

    parser.add_argument(
        '--index',
        type=int,
        required=True,
        help="SLURM array task ID"
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True, 
        help="Path to folder containing dat files"
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help="Path to save the reconstructions to"
    )
    parser.add_argument(
        '--sequence', 
        default='t2',
        type=str, 
        required=True,
        choices=['t2', 'dwi', 'both'],
        help="t2 or dwi or both"
    )
    args = vars(parser.parse_args())
    main(args)