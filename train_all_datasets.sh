#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
PRIOR_PROMPTS_PATH="${PRIOR_PROMPTS_PATH:-}"

DATASETS=(
  "datasets/1_shots_Anime_Faces"
  "datasets/5_shots_Anime_Faces"
  "datasets/5_flower_birdofparadise"
  "datasets/5_stanford_car"
)

METHODS=(
  "lora"
  "lora_prior"
  "dora"
)

EXTRA_ARGS=("$@")

if [[ -z "${PRIOR_PROMPTS_PATH}" ]]; then
  if [[ -f "${ROOT_DIR}/datasets/prior_prompts.txt" ]]; then
    PRIOR_PROMPTS_PATH="${ROOT_DIR}/datasets/prior_prompts.txt"
  elif [[ -f "${ROOT_DIR}/dataset/prior_prompts.txt" ]]; then
    PRIOR_PROMPTS_PATH="${ROOT_DIR}/dataset/prior_prompts.txt"
  fi
fi

for dataset_dir in "${DATASETS[@]}"; do
  dataset_name="$(basename "${dataset_dir}")"
  abs_dataset_dir="${ROOT_DIR}/${dataset_dir}"

  if [[ ! -d "${abs_dataset_dir}" ]]; then
    echo "[Skip] Missing dataset: ${abs_dataset_dir}"
    continue
  fi

  for method in "${METHODS[@]}"; do
    output_dir="${ROOT_DIR}/output/${dataset_name}/${method}"

    echo
    echo "[Run] dataset=${dataset_name} method=${method}"
    echo "[Info] output_dir=${output_dir}"

    rm -rf "${output_dir}"

    cmd=(
      "${PYTHON_BIN}"
      "${ROOT_DIR}/${TRAIN_SCRIPT}"
      "--method" "${method}"
      "--data_dir" "${abs_dataset_dir}"
      "--output_dir" "${output_dir}"
    )

    if [[ -n "${PRIOR_PROMPTS_PATH}" && -f "${PRIOR_PROMPTS_PATH}" ]]; then
      cmd+=("--prior_prompts_path" "${PRIOR_PROMPTS_PATH}")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi

    "${cmd[@]}"
  done
done

echo
echo "[Done] Finished training all datasets and methods."
