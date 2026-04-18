#!/bin/bash
#SBATCH --partition=data_mover
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=refresh_espirit_transfer
#SBATCH --output=logs_refresh_espirit/%x_%j.out
#SBATCH --error=logs_refresh_espirit/%x_%j.err

mkdir -p logs_refresh_espirit

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_kspace

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate/
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
STREAM_CONFIG="${STREAM_CONFIG:-config/streaming/refresh_espirit.yaml}"

python -m scripts.streaming.dwi.refresh_espirit_transfer --config "${STREAM_CONFIG}" --log-level INFO
