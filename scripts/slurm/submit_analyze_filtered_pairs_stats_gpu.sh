#!/bin/bash
#SBATCH --job-name=anlz_pairs_filt
#SBATCH --output=/scratch/pjbd103/sentences_alignment/logs/analyze_filtered_pairs_%j.out
#SBATCH --error=/scratch/pjbd103/sentences_alignment/logs/analyze_filtered_pairs_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail

source /scratch/pjbd103/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/scratch/pjbd103/sentences_alignment"
RUN_TAG="${RUN_TAG:-2026-02-28_fullCorpus_v1}"
LANGS="${LANGS:-en,es,fr,it,ca}"
MODELS="${MODELS:-bge,labse,sonar}"
SELECTION_MODE="${SELECTION_MODE:-best_f1}"
THRESHOLDS_CSV="${THRESHOLDS_CSV:-config/best_thresholds_strict_fullcorpus.csv}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${PROJECT_ROOT}/logs"

echo "== Analyze Filtered Pairs Stats =="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-manual}"
echo "RUN_TAG: ${RUN_TAG}"
echo "LANGS: ${LANGS}"
echo "MODELS: ${MODELS}"
echo "SELECTION_MODE: ${SELECTION_MODE}"
echo "THRESHOLDS_CSV: ${THRESHOLDS_CSV}"
echo "DEVICE: ${DEVICE}"
echo "BERTSCORE_BATCH_SIZE: ${BERTSCORE_BATCH_SIZE}"
echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

cd "${PROJECT_ROOT}"

python src/analyze_filtered_pairs_stats.py \
  --run-tag "${RUN_TAG}" \
  --languages "${LANGS}" \
  --models "${MODELS}" \
  --selection-mode "${SELECTION_MODE}" \
  --thresholds-csv "${THRESHOLDS_CSV}" \
  --device "${DEVICE}" \
  --bertscore-batch-size "${BERTSCORE_BATCH_SIZE}"

echo
echo "Job completado."
