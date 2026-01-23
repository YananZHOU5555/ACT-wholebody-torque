#!/usr/bin/env bash
set -e

# ============== [TORQUE EXTENSION] ==============
# ACT-wholebody training script - Train both torque variants
#
# This script trains two models sequentially:
#   1. USE_TORQUE=false (40000 steps) - baseline, should match original ACT
#   2. USE_TORQUE=true  (40000 steps) - with torque information
# ================================================

# Get script directory (ACT-wholebody/ACT-wholebody)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root (ACT-wholebody)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_FILE="${SCRIPT_DIR}/train.log"

# Common configuration
ACTION_DIM_OFFSET=3   # Skip first 3 dims (base velocity), use 14D arm action
STEPS=40000
SAVE_FREQ=20000

DATASET_NAME="ACT-100-wholebody"
DATASET_ROOT="${SCRIPT_DIR}/data/ACT-100-wholebody"

# Note: lerobot is installed from /home/zeno/ACT-wholebody/lerobot via pip install -e

# Function to train a model
train_model() {
    local USE_TORQUE=$1
    local OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/${DATASET_NAME}-torque-${USE_TORQUE}"
    local JOB_NAME="${DATASET_NAME}-torque-${USE_TORQUE}"

    echo ""
    echo "========================================"
    echo "Training ACT-wholebody"
    echo "  use_torque: ${USE_TORQUE}"
    echo "  steps: ${STEPS}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "========================================"
    echo ""

    accelerate launch \
      --multi_gpu \
      --num_processes=4 \
      --mixed_precision=fp16 \
      /workspace/ACT-wholebody-torque/lerobot/src/lerobot/scripts/lerobot_train.py \
      --policy.type=act \
      --dataset.repo_id=${DATASET_NAME} \
      --dataset.root=${DATASET_ROOT} \
      --dataset.video_backend=pyav \
      --output_dir "${OUTPUT_DIR}" \
      --job_name="${JOB_NAME}" \
      --policy.device=cuda \
      --policy.use_torque=${USE_TORQUE} \
      --policy.action_dim_offset=${ACTION_DIM_OFFSET} \
      --batch_size=32 \
      --steps=${STEPS} \
      --log_freq=100 \
      --eval_freq=${SAVE_FREQ} \
      --save_freq=${SAVE_FREQ} \
      --wandb.enable=false \
      --policy.repo_id=false 2>&1 | tee -a "${LOG_FILE}"
}

# ============================================
# Train Model 1: USE_TORQUE=false (baseline)
# ============================================
echo "============================================"
echo "  Step 1/2: Training USE_TORQUE=false"
echo "============================================"
train_model false

# ============================================
# Train Model 2: USE_TORQUE=true (with torque)
# ============================================
echo "============================================"
echo "  Step 2/2: Training USE_TORQUE=true"
echo "============================================"
train_model true

echo ""
echo "========================================"
echo "All training completed!"
echo "  Model 1: checkpoints/${DATASET_NAME}-torque-false"
echo "  Model 2: checkpoints/${DATASET_NAME}-torque-true"
echo "========================================"
