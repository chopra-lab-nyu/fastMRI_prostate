#!/bin/bash
#SBATCH --partition=cpu_short,cpu_medium,cpu_long
#SBATCH --time=1-00:00:00
#SBATCH --job-name=t2_recon
#SBATCH --export=ALL
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=[1-5]

source /gpfs/scratch/td2105/miniconda3/etc/profile.d/conda.sh
conda activate prostate_recon

cd /gpfs/data/chopralab/td2105/fastmri_prostate_internal/fastMRI_prostate

python3 -u fastmri_prostate_recon_from_dat.py \
    --index $SLURM_ARRAY_TASK_ID \
    --data_path /gpfs/data/chopralab/fastmri_prostate_dataset/data/sites/41stVida1/Hersh_VidaProstate \
    --output_path /gpfs/data/chopralab/fastmri_prostate_dataset/data/h5_data \
    --sequence t2
