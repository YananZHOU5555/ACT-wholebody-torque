#!/usr/bin/env bash
set -e

# ============== [FULLBODY TRAINING] ==============
# ACT-fullbody training script
# Config: use_base=true, use_torque=true
# ================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_FILE="${SCRIPT_DIR}/train.log"

# Training configuration
STEPS=40000
SAVE_FREQ=20000
LOG_FREQ=100

# Dataset configuration (using relative paths)
DATASET_NAME="ACT-final-V30"
DATASET_ROOT="${PROJECT_ROOT}/data/ACT-final-V30"

# Output configuration (using relative paths)
OUTPUT_DIR="${PROJECT_ROOT}/models/ACT-final-promax"
JOB_NAME="ACT-final-promax"

echo ""
echo "========================================"
echo "Training ACT-final-promax"
echo "  use_base: true"
echo "  use_torque: true"
echo "  steps: ${STEPS}"
echo "  save_freq: ${SAVE_FREQ}"
echo "  Dataset: ${DATASET_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"
echo ""

# Clear log file
echo "ACT-fullbody Training Log - $(date)" > "${LOG_FILE}"

# Set wandb API key (replace with your own key)
# export WANDB_API_KEY="your_wandb_api_key"

# Train fullbody model (single GPU)
python ${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.type act \
  --dataset.repo_id ${DATASET_NAME} \
  --dataset.root ${DATASET_ROOT} \
  --dataset.video_backend pyav \
  --output_dir "${OUTPUT_DIR}" \
  --job_name "${JOB_NAME}" \
  --policy.device cuda \
  --policy.use_base true \
  --policy.use_torque true \
  --policy.action_dim_offset 0 \
  --policy.repo_id "local/ACT-final-promax" \
  --policy.push_to_hub false \
  --batch_size 32 \
  --steps ${STEPS} \
  --log_freq ${LOG_FREQ} \
  --eval_freq ${SAVE_FREQ} \
  --save_freq ${SAVE_FREQ} \
  --wandb.enable true \
  --wandb.project "ACT-final-promax" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training completed!"
echo "  Model: ${OUTPUT_DIR}"
echo "========================================"
