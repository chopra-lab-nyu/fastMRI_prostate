#!/bin/bash
#SBATCH --partition=cpu_medium,cpu_long
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dwi_esc
#SBATCH --array=0-49
#SBATCH --output=logs_dwi_esc/%x_%A_%a.out
#SBATCH --error=logs_dwi_esc/%x_%A_%a.err

TOTAL_JOBS=50
DATA_DIR=/gpfs/data/prostatelab/jhad02/dwi_raw_dat_files
METADATA_CSV=/gpfs/data/prostatelab/jhad02/raw_data_csv_file/files_to_recall_full.csv
OUTPUT_DIR=/gpfs/scratch/td2105/dwi_esc_kspace_recon_all
PYTHON_SCRIPT=/gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/fastmri_prostate_recon_from_dat.py

mkdir -p "${OUTPUT_DIR}"

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_recon

python "${PYTHON_SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --metadata-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --directions b50x,b1000x \
    --averages 4:12 \
    --skip-metrics \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --job-count "${TOTAL_JOBS}"
