#!/bin/bash
#SBATCH --partition=data_mover
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=t2_transfer
#SBATCH --output=logs_streaming/%x_%j.out
#SBATCH --error=logs_streaming/%x_%j.err

mkdir -p logs_streaming

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_kspace

STREAM_CONFIG="${STREAM_CONFIG:-config/streaming_t2.yaml}"

python scripts/transfer_t2.py --config "${STREAM_CONFIG}" --log-level INFO
