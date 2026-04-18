#!/bin/bash
#SBATCH --partition=cpu_short,cpu_medium,cpu_long
#SBATCH --time=1-00:00:00
#SBATCH --job-name=t2_recon
#SBATCH --export=ALL
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH --output=logs_streaming/%x_%A_%a.out
#SBATCH --error=logs_streaming/%x_%A_%a.err
#SBATCH --array=0-4

mkdir -p logs_streaming

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_recon

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -u -m scripts.streaming.t2.recon_from_dat \
    --data-dir /gpfs/data/chopralab/fastmri_prostate_dataset/data/sites/41stVida1/Hersh_VidaProstate \
    --output-dir /gpfs/data/chopralab/fastmri_prostate_dataset/data/h5_data \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --job-count "${SLURM_ARRAY_TASK_COUNT:-5}"
