#!/bin/bash
#SBATCH --partition=cpu_medium,cpu_long
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dwi_esc
#SBATCH --array=0-49
#SBATCH --output=logs_streaming/%x_%A_%a.out
#SBATCH --error=logs_streaming/%x_%A_%a.err

DATA_DIR=/gpfs/data/prostatelab/jhad02/dwi_raw_dat_files
OUTPUT_DIR=/gpfs/scratch/td2105/dwi_esc_kspace_recon_all

mkdir -p logs_streaming
mkdir -p "${OUTPUT_DIR}"

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_recon

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m scripts.streaming.dwi.recon_from_dat \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --directions b50x,b1000x \
    --combines esc \
    --skip-metrics \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --job-count "${SLURM_ARRAY_TASK_COUNT:-50}"
