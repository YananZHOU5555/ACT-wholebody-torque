# ACT-wholebody-torque

ACT (Action Chunking Transformer) 策略的 torque 扩展版本，基于 LeRobot 框架。

## 快速开始

### 一键安装环境和数据

```bash
# 1. 克隆仓库
git clone https://github.com/YananZHOU5555/ACT-wholebody-torque.git
cd ACT-wholebody-torque

# 2. 安装 lerobot 环境 (推荐使用 conda)
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ./lerobot

# 3. 下载数据集
mkdir -p ACT-wholebody/data
huggingface-cli download B111ue/ACT-100-wholebody --repo-type dataset --local-dir ACT-wholebody/data/ACT-100-wholebody

# 4. 开始训练
cd ACT-wholebody
bash train.sh
```

## 项目结构

```
ACT-wholebody-torque/
├── ACT-wholebody/
│   ├── train.sh          # 训练脚本 (使用torchcodec加速)
│   ├── monitor.py        # 实时训练监控
│   └── bag2wholebody.py  # ROS bag转换脚本
├── lerobot/              # 修改后的lerobot（含torque扩展）
└── README.md
```

## 核心修改

### 1. 新增配置参数 (`lerobot/src/lerobot/policies/act/configuration_act.py`)

```python
use_torque: bool = False      # True: 使用effort数据; False: 使用全零
action_dim_offset: int = 0    # 跳过action前N维 (如: 3跳过base velocity)
```

### 2. 模型修改 (`lerobot/src/lerobot/policies/act/modeling_act.py`)

- 添加 torque projection layer (14D → dim_model)
- Transformer encoder 输入: `[latent, state, torque, image_features]`
- 支持 action 维度裁剪 (17D → 14D)

## 训练配置 (train.sh)

- **视频后端**: torchcodec (比pyav快约7倍)
- **步数**: 40000 steps × 2 (torque=false, torque=true)
- **保存频率**: 每20000步保存并上传到HuggingFace
- **Batch size**: 32
- **GPU**: 支持多GPU训练 (默认4卡)
- **Action维度**: 14D (跳过前3D base velocity)
- **输出目录**: `checkpoints/ACT-100-wholebody-torque-{false,true}`

## 数据说明

| 字段 | 维度 | 说明 |
|------|------|------|
| observation.state | 14D | 双臂关节位置 |
| observation.effort | 14D | 双臂关节力矩 |
| action | 17D | 3D base + 14D arms (训练时只用后14D) |

## 训练好的模型

| 模型 | HuggingFace 链接 | 说明 |
|------|------------------|------|
| torque=false | [B111ue/ACT-100-wholebody-torque-false](https://huggingface.co/B111ue/ACT-100-wholebody-torque-false) | 基线模型，不使用力矩信息 |
| torque=true | [B111ue/ACT-100-wholebody-torque-true](https://huggingface.co/B111ue/ACT-100-wholebody-torque-true) | 使用力矩信息的模型 |

## 相关资源

- **数据集**: [B111ue/ACT-100-wholebody](https://huggingface.co/datasets/B111ue/ACT-100-wholebody)
- **代码仓库**: [YananZHOU5555/ACT-wholebody-torque](https://github.com/YananZHOU5555/ACT-wholebody-torque)
