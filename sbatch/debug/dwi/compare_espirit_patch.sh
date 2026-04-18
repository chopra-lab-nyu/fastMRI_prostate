#!/bin/bash

#SBATCH --partition=radiology,cpu_medium,cpu_long
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=espirit_cmp
#SBATCH --output=logs_debug/%x_%A_%a.out
#SBATCH --error=logs_debug/%x_%A_%a.err

ACCESSIONS_FILE="${ACCESSIONS_FILE:-/gpfs/scratch/td2105/dwi_stream/test_accessions_esprit_patch.txt}"
LABELS_CSV="${LABELS_CSV:-/gpfs/data/prostatelab/processed_data/csv/prostate_mri_radiology_reports_201212_202508.csv}"
H5_ROOT="${H5_ROOT:-/gpfs/scratch/td2105/dwi_stream/recons_rss_espirit}"
DAT_DIR="${DAT_DIR:-/gpfs/scratch/td2105/dwi_stream/temp_dat_files}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/td2105/dwi_stream/espirit_patch_compare}"

mkdir -p logs_debug
mkdir -p "${OUTPUT_DIR}"

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
module add gcc12/12.2.0
conda activate prostate_kspace

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="/tmp"
export XDG_CACHE_HOME="/tmp"

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

JOB_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
JOB_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

python -m scripts.debug.dwi.compare_espirit_patch \
    --labels-csv "${LABELS_CSV}" \
    --h5-root "${H5_ROOT}" \
    --dat-dir "${DAT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --accessions-file "${ACCESSIONS_FILE}" \
    --scheme "b50_4_b1000_12" \
    --job-count "${JOB_COUNT}" \
    --job-index "${JOB_INDEX}"
