#!/bin/bash
#SBATCH --partition=radiology,cpu_medium,cpu_long
#SBATCH --time=6-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-39
#SBATCH --job-name=dwi_direct_worker
#SBATCH --output=logs_dwi_direct_state_bank/%x_%A_%a.out
#SBATCH --error=logs_dwi_direct_state_bank/%x_%A_%a.err

set -euo pipefail

mkdir -p logs_dwi_direct_state_bank

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
module add gcc12/12.2.0
conda activate prostate_kspace
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
STREAM_CONFIG="${STREAM_CONFIG:-config/streaming/dwi_direct_state_bank.yaml}"

python -u -m scripts.streaming.dwi.direct_state_bank_worker \
    --config "${STREAM_CONFIG}" \
    --worker-id "${SLURM_ARRAY_TASK_ID}" \
    --log-level INFO
