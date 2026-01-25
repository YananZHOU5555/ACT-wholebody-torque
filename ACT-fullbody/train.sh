#!/usr/bin/env bash
set -e

# ============== [FULLBODY EXTENSION] ==============
# ACT-fullbody training script - Train 2 configurations
#
# This script trains two models sequentially:
#   1. base-only:    use_base=true,  use_torque=false
#   2. fullbody:     use_base=true,  use_torque=true
# ================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_FILE="${SCRIPT_DIR}/train.log"

# Common configuration
STEPS=40000
SAVE_FREQ=20000
LOG_FREQ=100

DATASET_NAME="ACT-120-V30"
DATASET_ROOT="${SCRIPT_DIR}/data/ACT-120-V30"

# HuggingFace token (set via environment variable or huggingface-cli login)
# export HF_TOKEN="your_token_here"

# Function to train a model
train_model() {
    local USE_BASE=$1
    local USE_TORQUE=$2
    local CONFIG_NAME=$3
    local OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/ACT-fullbody-${CONFIG_NAME}"
    local JOB_NAME="ACT-fullbody-${CONFIG_NAME}"
    local HF_REPO_ID="B111ue/ACT-fullbody-${CONFIG_NAME}"

    echo ""
    echo "========================================"
    echo "Training ACT-fullbody: ${CONFIG_NAME}"
    echo "  use_base: ${USE_BASE}"
    echo "  use_torque: ${USE_TORQUE}"
    echo "  steps: ${STEPS}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  HF Repo: ${HF_REPO_ID}"
    echo "========================================"
    echo ""

    accelerate launch \
      --multi_gpu \
      --num_processes=4 \
      --mixed_precision=fp16 \
      ${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py \
      --policy.type act \
      --dataset.repo_id ${DATASET_NAME} \
      --dataset.root ${DATASET_ROOT} \
      --dataset.video_backend torchcodec \
      --output_dir "${OUTPUT_DIR}" \
      --job_name "${JOB_NAME}" \
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
      --wandb.project "ACT-fullbody-120" \
      --policy.repo_id ${HF_REPO_ID} \
      --policy.push_to_hub true 2>&1 | tee -a "${LOG_FILE}"

    echo "Completed: ${CONFIG_NAME}"
    echo ""
}

# Clear log file
echo "ACT-fullbody Training Log - $(date)" > "${LOG_FILE}"

# ============================================
# Train Model 1: base-only (use_base=true, use_torque=false)
# ============================================
echo "============================================"
echo "  Step 1/2: Training base-only"
echo "============================================"
train_model true false "base-only"

# ============================================
# Train Model 2: fullbody (use_base=true, use_torque=true)
# ============================================
echo "============================================"
echo "  Step 2/2: Training fullbody"
echo "============================================"
train_model true true "fullbody"

# ============================================
# Generate loss comparison plot
# ============================================
echo "============================================"
echo "  Generating loss comparison plot..."
echo "============================================"
python "${SCRIPT_DIR}/plot_loss_comparison.py"

echo ""
echo "========================================"
echo "All training completed!"
echo "  Model 1: checkpoints/ACT-fullbody-base-only"
echo "  Model 2: checkpoints/ACT-fullbody-fullbody"
echo ""
echo "Loss comparison plot: loss_comparison.png"
echo "========================================"
