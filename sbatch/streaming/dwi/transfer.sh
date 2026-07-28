#!/bin/bash
#SBATCH --partition=data_mover
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --job-name=dwi_transfer
#SBATCH --output=logs_streaming/%x_%A_%a.out
#SBATCH --error=logs_streaming/%x_%A_%a.err

set -euo pipefail

mkdir -p logs_streaming

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_kspace
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
STREAM_CONFIG="${STREAM_CONFIG:-config/streaming/dwi.yaml}"

python -u -m scripts.streaming.dwi.transfer \
    --config "${STREAM_CONFIG}" \
    --worker-id "${SLURM_ARRAY_TASK_ID}" \
    --log-level INFO
