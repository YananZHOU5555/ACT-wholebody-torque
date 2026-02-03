# ACT-wholebody-torque

ACT (Action Chunking Transformer) whole-body control extension based on LeRobot framework, supporting coordinated control of dual-arm robot + mobile base.

## Features

- 17D whole-body control: 3D base velocity + 14D dual-arm joints
- Torque information fusion: utilizing joint torque to improve control precision
- 3-camera visual input: top + left wrist + right wrist

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/YananZHOU5555/ACT-wholebody-torque.git
cd ACT-wholebody-torque

# Create conda environment
conda create -n lerobot python=3.10 -y
conda activate lerobot

# Install dependencies
pip install -e ./lerobot
```

### 2. Download Models

```bash
# Demo 1: Pick and Place
huggingface-cli download B111ue/ACT-pick-place --local-dir models/ACT-pick-place

# Demo 2: Hanger Grasping
huggingface-cli download B111ue/ACT-hanger --local-dir models/ACT-hanger

# Demo 3: Fold Clothes
huggingface-cli download B111ue/ACT-fold-clothes --local-dir models/ACT-fold-clothes
```

### 3. Run Demo

```bash
conda activate lerobot

# Demo 1: Pick and Place
python demos/demo1_pick_place/run.py

# Demo 2: Hanger Grasping
python demos/demo2_hanger/run.py

# Demo 3: Fold Clothes
python demos/demo3_fold_clothes/run.py
```

## Demo Description

| Demo | Task | Model | Data Size |
|------|------|------|--------|
| demo1_pick_place | Place basket on red marker, pick yellow pepper into basket | ACT-pick-place | 100 dual-arm + 20 whole-body |
| demo2_hanger | Grasp hanger, pass to right hand, throw into basket | ACT-hanger | 100 whole-body |
| demo3_fold_clothes | Fold clothes into box, transport to shelf | ACT-fold-clothes | 50 whole-body |

## Project Structure

```
ACT-wholebody-torque/
├── demos/
│   ├── demo1_pick_place/
│   │   └── run.py              # Pick and Place deployment script
│   ├── demo2_hanger/
│   │   └── run.py              # Hanger Grasping deployment script
│   └── demo3_fold_clothes/
│       └── run.py              # Fold Clothes deployment script
├── models/                     # Model weights (need to download)
│   ├── ACT-pick-place/
│   ├── ACT-hanger/
│   └── ACT-fold-clothes/
├── scripts/
│   ├── train.sh                # Training script
│   └── data_convert/           # Data conversion tools
├── lerobot/                    # Modified LeRobot framework
└── README.md
```

## ROS Topics

### Input
| Topic | Type | Description |
|-------|------|------|
| `/realsense_top/color/image_raw/compressed` | CompressedImage | Top camera |
| `/realsense_left/color/image_raw/compressed` | CompressedImage | Left wrist camera |
| `/realsense_right/color/image_raw/compressed` | CompressedImage | Right wrist camera |
| `/robot/arm_left/joint_states_single` | JointState | Left arm joint states (position, velocity, effort) |
| `/robot/arm_right/joint_states_single` | JointState | Right arm joint states (position, velocity, effort) |
| `/ranger_base_node/odom` | Odometry | Base odometry |

### Output
| Topic | Type | Description |
|-------|------|------|
| `/robot/arm_left/vla_joint_cmd` | JointState | Left arm joint command |
| `/robot/arm_right/vla_joint_cmd` | JointState | Right arm joint command |
| `/cmd_vel` | Twist | Base velocity command |

## Command Line Arguments

All demo scripts support the following arguments:

```bash
python demos/demo1_pick_place/run.py \
    --ckpt /path/to/model       # Specify model path
    --rate 10                   # Control frequency Hz (default: 10)
    --smoothing 0.3             # EMA smoothing coefficient (default: 0.3)
    --no-smoothing              # Disable EMA smoothing
```

## Train Your Own Model

### 1. Data Conversion

Convert ROS bag to LeRobot format:

```bash
python scripts/data_convert/bag2wholebody_final.py
```

### 2. Training

```bash
bash scripts/train.sh
```

## Core Configuration

```python
# lerobot/src/lerobot/policies/act/configuration_act.py

use_torque: bool = True       # Use torque information
use_base: bool = True         # Use base velocity information
action_dim_offset: int = 0    # Action dimension offset
```

## Data Format

| Field | Dimension | Description |
|------|------|------|
| action | 17D | [base_vx, base_vy, base_omega, left_joints x7, right_joints x7] |
| observation.state | 14D | [left_joints x7, right_joints x7] |
| observation.effort | 14D | [left_effort x7, right_effort x7] |
| observation.base_velocity | 3D | [base_vx, base_vy, base_omega] |
| observation.images.* | 224x224x3 | RGB images from 3 cameras (top, left wrist, right wrist) |

## License

Apache 2.0
