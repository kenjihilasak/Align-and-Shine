#!/bin/bash
#SBATCH --job-name=align_full_sonar
#SBATCH --output=/scratch/pjbd103/sentences_alignment/logs/align_full_sonar_%j.out
#SBATCH --error=/scratch/pjbd103/sentences_alignment/logs/align_full_sonar_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail

source /scratch/pjbd103/miniconda3/etc/profile.d/conda.sh
conda activate SentAlignSONAR

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/scratch/pjbd103/sentences_alignment"
RUN_TAG="2026-02-28_fullCorpus_v1"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_TAG}"
MODEL_KEY="sonar"
MODEL_DIR="${PROJECT_ROOT}/SentAlign_SONAR"
OUT_REL="../01_raw_alignments/${MODEL_KEY}"
LANGS=(en es fr it ca)
NPROC="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${RUN_ROOT}"

declare -A SONAR_LANG
SONAR_LANG[en]="eng_Latn"
SONAR_LANG[es]="spa_Latn"
SONAR_LANG[fr]="fra_Latn"
SONAR_LANG[it]="ita_Latn"
SONAR_LANG[ca]="cat_Latn"

echo "== SONAR Full-Corpus Alignment =="
echo "Run root: ${RUN_ROOT}"
echo "Model dir: ${MODEL_DIR}"
echo "Languages: ${LANGS[*]}"
echo "Num proc: ${NPROC}"

for lang in "${LANGS[@]}"; do
  echo
  echo "==== [${lang}] starting (${MODEL_KEY}) ===="

  wiki_src="${PROJECT_ROOT}/data/${lang}/wiki"
  viki_src="${PROJECT_ROOT}/data/${lang}/viki"
  if [[ ! -d "${wiki_src}" || ! -d "${viki_src}" ]]; then
    echo "[ERROR] Missing input dirs for ${lang}: ${wiki_src} or ${viki_src}" >&2
    exit 1
  fi

  sonar_code="${SONAR_LANG[$lang]:-eng_Latn}"
  lang_root="${RUN_ROOT}/${lang}"
  work_dir="${lang_root}/_work_${MODEL_KEY}"
  out_dir="${lang_root}/01_raw_alignments/${MODEL_KEY}"
  log_dir="${lang_root}/logs"
  lang_log="${log_dir}/${MODEL_KEY}_${SLURM_JOB_ID:-manual}.log"

  mkdir -p "${work_dir}" "${out_dir}" "${log_dir}"
  ln -sfn "${wiki_src}" "${work_dir}/wiki"
  ln -sfn "${viki_src}" "${work_dir}/viki"

  (
    cd "${MODEL_DIR}"
    python files2align.py -dir "${work_dir}" -sl wiki -out "${OUT_REL}"
    python sentAlign.py \
      -dir "${work_dir}" \
      -sl wiki \
      -tl viki \
      -slsonar "${sonar_code}" \
      -tlsonar "${sonar_code}" \
      -out "${OUT_REL}" \
      -device cuda \
      -proc "${NPROC}"
  ) 2>&1 | tee "${lang_log}"

  echo "==== [${lang}] completed (${MODEL_KEY}) ===="
  echo "Outputs: ${out_dir}"
done

echo
echo "SONAR full-corpus job completed."
