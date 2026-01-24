# ACT-wholebody 部署脚本

ACT (Action Chunking with Transformers) 策略部署脚本，用于 Piper 双臂机器人。

## 环境要求

- Python 3.10+
- ROS (已测试 ROS Noetic)
- CUDA (推荐)
- lerobot 环境

## 安装

```bash
# 激活 lerobot conda 环境
conda activate lerobot

# 确保 lerobot 已安装
cd /home/zeno/ACT-wholebody/lerobot
pip install -e .
```

## 模型

支持两个预训练模型：

| 模型 | 路径 | 说明 |
|------|------|------|
| torque-true | `checkpoints/ACT-100-wholebody-torque-true` | 使用力矩信息 |
| torque-false | `checkpoints/ACT-100-wholebody-torque-false` | 不使用力矩信息 (基线) |

## 使用方法

### 基本运行

```bash
conda activate lerobot
cd /home/zeno/ACT-wholebody/ACT-wholebody

# 使用力矩信息 (默认)
python run.py --model torque-true

# 不使用力矩信息
python run.py --model torque-false
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `torque-true` | 模型选择: `torque-true`, `torque-false`, `local` |
| `--ckpt` | None | 本地 checkpoint 路径 (仅 `--model local` 时使用) |
| `--rate` | 10.0 | 控制频率 (Hz) |
| `--smoothing` | 0.3 | EMA 平滑系数 (0-1) |
| `--no-smoothing` | False | 禁用 EMA 平滑 |

### 示例

```bash
# 使用自定义控制频率
python run.py --model torque-true --rate 20

# 调整平滑系数
python run.py --model torque-false --smoothing 0.5

# 禁用平滑
python run.py --no-smoothing

# 使用本地自定义模型
python run.py --model local --ckpt /path/to/your/checkpoint
```

## ROS Topics

### 订阅 (Subscribers)

| Topic | 类型 | 说明 |
|-------|------|------|
| `/realsense_top/color/image_raw/compressed` | CompressedImage | 主相机 |
| `/realsense_left/color/image_raw/compressed` | CompressedImage | 左腕相机 |
| `/realsense_right/color/image_raw/compressed` | CompressedImage | 右腕相机 |
| `/robot/arm_left/joint_states_single` | JointState | 左臂关节状态 |
| `/robot/arm_right/joint_states_single` | JointState | 右臂关节状态 |
| `/robot/arm_left/end_pose` | PoseStamped | 左臂末端位姿 |
| `/robot/arm_right/end_pose` | PoseStamped | 右臂末端位姿 |
| `/robot/base/velocity` | Twist | 底盘速度 |

### 发布 (Publishers)

| Topic | 类型 | 说明 |
|-------|------|------|
| `/robot/arm_left/vla_joint_cmd` | JointState | 左臂关节命令 |
| `/robot/arm_right/vla_joint_cmd` | JointState | 右臂关节命令 |

## 模型输入输出

### 输入
- **State**: 14D (左臂 7D + 右臂 7D) - 关节位置
- **Effort**: 14D (左臂 7D + 右臂 7D) - 关节力矩 (仅 torque-true)
- **Images**: 4 个相机, 224x224, ImageNet 归一化

### 输出
- **Action**: 14D (左臂 7D + 右臂 7D) - 目标关节位置

## 调试

```bash
# 查看相关 topics
rostopic list | grep -E "(arm_left|arm_right|realsense)"

# 检查关节状态
rostopic echo /robot/arm_left/joint_states_single

# 检查命令输出
rostopic echo /robot/arm_left/vla_joint_cmd
```
