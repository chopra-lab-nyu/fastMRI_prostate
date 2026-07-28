#!/bin/bash
#SBATCH --partition=radiology,cpu_medium,cpu_long
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dwi_espirit_state_bank_v2
#SBATCH --array=0-39
#SBATCH --output=logs_build_espirit_state_bank_v2/%x_%A_%a.out
#SBATCH --error=logs_build_espirit_state_bank_v2/%x_%A_%a.err

mkdir -p logs_build_espirit_state_bank_v2

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
module add gcc12/12.2.0
conda activate prostate_kspace

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
STREAM_CONFIG="${STREAM_CONFIG:-config/streaming/dwi_build_espirit_state_bank.yaml}"

python -u -m scripts.streaming.dwi.build_espirit_state_bank_h5 \
    --config "${STREAM_CONFIG}" \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --job-count "${SLURM_ARRAY_TASK_COUNT:-40}"
