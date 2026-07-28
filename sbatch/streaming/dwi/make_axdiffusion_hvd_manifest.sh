#!/bin/bash
#SBATCH --partition=data_mover
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=dwi_axhvd_manifest
#SBATCH --output=logs_dwi_direct_state_bank/%x_%j.out
#SBATCH --error=logs_dwi_direct_state_bank/%x_%j.err

set -euo pipefail

mkdir -p logs_dwi_direct_state_bank

SOURCE_ROOT="${SOURCE_ROOT:-/mnt/td2105/MRIScan/Archive/yarra_rds}"
OUT="${OUT:-/gpfs/data/prostatelab/processed_data/csv/kspace_prostate_dwi_file_metadata_yarra_axdiffusion_hvd.csv}"
TMP="${OUT}.tmp.${SLURM_JOB_ID:-$$}"

mkdir -p "$(dirname "${OUT}")"
cd "${SOURCE_ROOT}"

{
    echo "size,path"
    find . -type f -path "./*/Hersh_VidaProstateDiffusion/*#AXDIFFUSION_HVD.dat" -exec du -h {} + \
        | sort -k2 \
        | sed $'s/\t/,/1'
} > "${TMP}"

mv "${TMP}" "${OUT}"
echo "Wrote ${OUT}"
