#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="/workspace/ACT-wholebody-torque/ACT-fullbody/data"
CHECKPOINT_ROOT="/workspace/ACT-wholebody-torque/ACT-fullbody/checkpoints"

STEPS=20000
SAVE_FREQ=20000
LOG_FREQ=500

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export HF_ENDPOINT="https://hf-mirror.com"

# fullbody first, then others
CONFIGS=(
  "fullbody:true:true"
  "arm-only:false:false"
  "arm-torque:false:true"
  "base-only:true:false"
)

train_single_model() {
  local DATASET_NAME="$1"
  local CHECKPOINT_DIR="$2"
  local WANDB_PROJECT="$3"
  local HF_REPO="$4"
  local NAME="$5"
  local USE_BASE="$6"
  local USE_TORQUE="$7"

  echo ">> Training ${NAME}"

  accelerate launch --quiet \
    --multi_gpu --num_processes=4 --mixed_precision=fp16 \
    ${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py \
    --policy.type act \
    --policy.repo_id "${HF_REPO}-${NAME}" \
    --dataset.repo_id ${DATASET_NAME} \
    --dataset.root ${DATA_ROOT}/${DATASET_NAME} \
    --dataset.video_backend torchcodec \
    --output_dir "${CHECKPOINT_DIR}/${NAME}" \
    --job_name "${DATASET_NAME}-${NAME}" \
    --policy.device cuda \
    --policy.use_base ${USE_BASE} \
    --policy.use_torque ${USE_TORQUE} \
    --policy.action_dim_offset 0 \
    --batch_size 32 \
    --steps ${STEPS} \
    --log_freq ${LOG_FREQ} \
    --eval_freq ${SAVE_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --wandb.enable true \
    --wandb.project "${WANDB_PROJECT}" 2>&1 | grep -v "^\[rank" | grep -v "^Warning" | grep -v "UnsupportedFieldAttributeWarning"

  echo ">> Done training: ${NAME}"

  # Upload immediately after training
  echo ">> Uploading ${NAME} to HuggingFace: ${HF_REPO}-${NAME}"
  huggingface-cli upload "${HF_REPO}-${NAME}" "${CHECKPOINT_DIR}/${NAME}" . --repo-type model
  echo ">> Upload complete: https://huggingface.co/${HF_REPO}-${NAME}"
}

train_all_models() {
  local DATASET_NAME="$1"
  local CHECKPOINT_DIR="$2"
  local WANDB_PROJECT="$3"
  local HF_REPO="$4"

  echo "========================================"
  echo "Dataset: ${DATASET_NAME}"
  echo "Checkpoint: ${CHECKPOINT_DIR}"
  echo "========================================"

  mkdir -p "${CHECKPOINT_DIR}"

  for CONFIG in "${CONFIGS[@]}"; do
    IFS=":" read -r NAME USE_BASE USE_TORQUE <<< "$CONFIG"
    train_single_model "${DATASET_NAME}" "${CHECKPOINT_DIR}" "${WANDB_PROJECT}" "${HF_REPO}" "${NAME}" "${USE_BASE}" "${USE_TORQUE}"
  done

  echo ">> All 4 models for ${DATASET_NAME} completed and uploaded!"
}

# Train ACT-20-V30-fixed first (fullbody first)
train_all_models \
  "ACT-20-V30-fixed" \
  "${CHECKPOINT_ROOT}/ACT-20-V30-fixed-new" \
  "ACT-20-V30-new" \
  "B111ue/ACT-20-V30-fixed-new"

# Train ACT-120-V30-fixed second (fullbody first)
train_all_models \
  "ACT-120-V30-fixed" \
  "${CHECKPOINT_ROOT}/ACT-120-V30-fixed-new" \
  "ACT-120-V30-new" \
  "B111ue/ACT-120-V30-fixed-new"

echo "========================================"
echo "ALL DONE!"
echo "========================================"
