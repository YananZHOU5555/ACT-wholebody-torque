#!/usr/bin/env bash
set -e

cd /workspace/ACT-wholebody-torque/ACT-fullbody
source /venv/main/bin/activate

export HF_ENDPOINT="https://hf-mirror.com"

echo "========================================"
echo "Step 1: Convert ACT-120-V30-fixed"
echo "========================================"
rm -rf /workspace/ACT-wholebody-torque/ACT-fullbody/data/ACT-120-V30-fixed
python bag2wholebody_120_fixed.py

echo "========================================"
echo "Step 2: Upload ACT-120-V30-fixed to HF"
echo "========================================"
huggingface-cli upload B111ue/ACT-120-V30-fixed /workspace/ACT-wholebody-torque/ACT-fullbody/data/ACT-120-V30-fixed . --repo-type dataset
echo "Uploaded: https://huggingface.co/datasets/B111ue/ACT-120-V30-fixed"

echo "========================================"
echo "Step 3: Convert ACT-20-V30-fixed"
echo "========================================"
rm -rf /workspace/ACT-wholebody-torque/ACT-fullbody/data/ACT-20-V30-fixed
python bag2wholebody_20_fixed.py

echo "========================================"
echo "Step 4: Upload ACT-20-V30-fixed to HF"
echo "========================================"
huggingface-cli upload B111ue/ACT-20-V30-fixed /workspace/ACT-wholebody-torque/ACT-fullbody/data/ACT-20-V30-fixed . --repo-type dataset
echo "Uploaded: https://huggingface.co/datasets/B111ue/ACT-20-V30-fixed"

echo "========================================"
echo "ALL DONE!"
echo "========================================"
