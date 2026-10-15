#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DATASET=${DATASET:-ISIC}
SHOT=${SHOT:-5}
QUERY=${QUERY:-2}
GPU=${GPU:-0}
MAX_EPOCH=${MAX_EPOCH:-200}
SAMPLER_BATCH=${SAMPLER_BATCH:-200}
MODEL_LIST=${MODEL_LIST:-resnet12}
TRAIN_SEEDS=${TRAIN_SEEDS:-"1 42 332 1116 1999"}
TEST_SEEDS=${TEST_SEEDS:-"1 42 332 1116 1999"}
LR_LIST=${LR_LIST:-5e-4}
MLP_LR_LIST=${MLP_LR_LIST:-1e-2}
C_LIST=${C_LIST:-}
TEST_EPISODES=${TEST_EPISODES:-2000}
CHECKPOINT=${CHECKPOINT:-max_acc.pth}
SKIP_TRAIN=${SKIP_TRAIN:-0}
RESULTS_CSV=${RESULTS_CSV:-./results/${DATASET}_results.csv}
SUMMARY_CSV=${SUMMARY_CSV:-./results/${DATASET}_summary.csv}
SAVE_PATH=${SAVE_PATH:-}

if [[ -z "$C_LIST" ]]; then
  if [[ "$SHOT" == "1" ]]; then
    C_LIST=1e-3
  elif [[ "$SHOT" == "5" ]]; then
    C_LIST=5e-3
  else
    echo "Set C_LIST explicitly for SHOT=$SHOT" >&2
    exit 2
  fi
fi

case "$DATASET" in
  ISIC)
    WAY=${WAY:-3}
    VALIDATION_WAY=${VALIDATION_WAY:-2}
    : "${ISIC_IMAGE_PATH:?Set ISIC_IMAGE_PATH to the ISIC image directory}"
    ISIC_SPLIT_PATH=${ISIC_SPLIT_PATH:-./data/ISIC/split}
    DATA_ARGS=(--isic_image_path "$ISIC_IMAGE_PATH" --isic_split_path "$ISIC_SPLIT_PATH")
    ;;
  SD198)
    WAY=${WAY:-5}
    VALIDATION_WAY=${VALIDATION_WAY:-4}
    : "${SD198_IMAGE_PATH:?Set SD198_IMAGE_PATH to the SD-198 image directory}"
    SD198_SPLIT_PATH=${SD198_SPLIT_PATH:-./data/SD198/split}
    DATA_ARGS=(--sd198_image_path "$SD198_IMAGE_PATH" --sd198_split_path "$SD198_SPLIT_PATH")
    ;;
  RFMiD)
    WAY=${WAY:-5}
    VALIDATION_WAY=${VALIDATION_WAY:-5}
    : "${RFMID_IMAGE_PATH:?Set RFMID_IMAGE_PATH to the RFMiD dataset root}"
    RFMID_SPLIT_PATH=${RFMID_SPLIT_PATH:-./data/RFMiD/split}
    DATA_ARGS=(--rfmid_image_path "$RFMID_IMAGE_PATH" --rfmid_split_path "$RFMID_SPLIT_PATH")
    ;;
  *)
    echo "Unsupported DATASET=$DATASET (choose ISIC, SD198, or RFMiD)" >&2
    exit 2
    ;;
esac

FLAGS=()
FLAGS+=(--hyperbolic --not-riemannian)
[[ "$SKIP_TRAIN" == "1" ]] && FLAGS+=(--skip_train)
read -r -a TRAIN_SEED_LIST <<< "$TRAIN_SEEDS"

for MODEL in $MODEL_LIST; do
  for LR in $LR_LIST; do
    for MLP_LR in $MLP_LR_LIST; do
      for C in $C_LIST; do
        for TRAIN_SEED in "${TRAIN_SEED_LIST[@]}"; do
          RUN_FLAGS=("${FLAGS[@]}")
          if [[ -n "$SAVE_PATH" ]]; then
            RUN_SAVE_PATH="$SAVE_PATH"
            if [[ ${#TRAIN_SEED_LIST[@]} -gt 1 ]]; then
              RUN_SAVE_PATH="${SAVE_PATH%/}/seed${TRAIN_SEED}"
              [[ "$SKIP_TRAIN" == "1" ]] || mkdir -p "$SAVE_PATH"
            fi
            RUN_FLAGS+=(--save_path "$RUN_SAVE_PATH")
          fi
          CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" run_experiment.py \
            --dataset "$DATASET" \
            --gpu 0 \
            --max_epoch "$MAX_EPOCH" \
            --sampler_batch "$SAMPLER_BATCH" \
            --query "$QUERY" \
            --way "$WAY" \
            --validation_way "$VALIDATION_WAY" \
            --shot "$SHOT" \
            --lr "$LR" \
            --mlp_lr "$MLP_LR" \
            --model "$MODEL" \
            --c "$C" \
            --seed "$TRAIN_SEED" \
            --test_seeds "$TEST_SEEDS" \
            --test_episodes "$TEST_EPISODES" \
            --checkpoint "$CHECKPOINT" \
            --results_csv "$RESULTS_CSV" \
            --summary_csv "$SUMMARY_CSV" \
            "${DATA_ARGS[@]}" \
            "${RUN_FLAGS[@]}"
        done
      done
    done
  done
done
