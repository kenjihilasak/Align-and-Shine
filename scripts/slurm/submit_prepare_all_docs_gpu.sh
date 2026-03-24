#!/bin/bash
#SBATCH --job-name=prepare_all_docs
#SBATCH --output=/scratch/pjbd103/sentences_alignment/logs/prepare_%j.out
#SBATCH --error=/scratch/pjbd103/sentences_alignment/logs/prepare_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

source /scratch/pjbd103/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /scratch/pjbd103/sentences_alignment
mkdir -p logs

python src/prepare_sentalign_data.py \
  --langs es,en,fr,it,ca \
  --all-docs \
  --clean-output \
  --gpu-check
