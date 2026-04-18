#!/bin/bash

# Override the array range at submit time for the current dataset size.
# Example for the 7-file mini_dataset: sbatch --array=0-6 sbatch/debug/dwi/compare_public_h5_recon_array.sh
#SBATCH --partition=radiology,cpu_medium,cpu_long
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-0
#SBATCH --job-name=public_h5_cmp
#SBATCH --output=logs_debug/%x_%A_%a.out
#SBATCH --error=logs_debug/%x_%A_%a.err

INPUT_DIR="${INPUT_DIR:-/gpfs/data/chopra_public/fastmri_prostate_dataset/data/mini_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/td2105/dwi_stream/public_h5_dwi_recon_compare}"

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

cmd=(
    python -m scripts.debug.dwi.compare_public_h5_dwi_recon
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --job-count "${JOB_COUNT}"
    --job-index "${JOB_INDEX}"
)

"${cmd[@]}"
