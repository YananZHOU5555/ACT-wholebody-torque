# ACT-wholebody-torque

ACT (Action Chunking Transformer) 策略的全身控制扩展版本，基于 LeRobot 框架。

支持两个子项目：
- **ACT-wholebody**：14D 手臂控制 + 力矩信息
- **ACT-fullbody**：17D 全身控制（底盘 + 手臂）+ 力矩信息

## 快速开始

### 1. 安装环境

```bash
# 克隆仓库
git clone https://github.com/YananZHOU5555/ACT-wholebody-torque.git
cd ACT-wholebody-torque

# 安装 lerobot 环境 (推荐使用 conda)
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ./lerobot
```

### 2. 下载数据集

```bash
mkdir -p ACT-wholebody/data
huggingface-cli download B111ue/ACT-100-wholebody --repo-type dataset --local-dir ACT-wholebody/data/ACT-100-wholebody
```

### 3. 选择训练模式

#### 方式A：ACT-wholebody（14D 手臂控制）
```bash
cd ACT-wholebody
bash train.sh
```

#### 方式B：ACT-fullbody（17D 全身控制）
```bash
cd ACT-fullbody
bash train.sh
```

## 项目结构

```
ACT-wholebody-torque/
├── ACT-wholebody/              # 14D 手臂控制子项目
│   ├── train.sh                # 训练脚本 (torque=false/true)
│   ├── run.py                  # 部署脚本
│   └── data/                   # 数据集
│
├── ACT-fullbody/               # 17D 全身控制子项目 (NEW)
│   ├── train.sh                # 训练脚本 (4种配置)
│   ├── plot_loss_comparison.py # Loss对比图生成
│   └── data/                   # 数据集软链接
│
├── lerobot/                    # 修改后的lerobot框架
│   └── src/lerobot/policies/act/
│       ├── configuration_act.py  # 配置参数
│       └── modeling_act.py       # 模型实现
│
└── README.md
```

## 两个子项目对比

| 特性 | ACT-wholebody | ACT-fullbody |
|------|---------------|--------------|
| Action维度 | 14D (只有手臂) | 17D (底盘+手臂) |
| State维度 | 14D | 17D |
| action_dim_offset | 3 (跳过base) | 0 (完整action) |
| 训练配置数 | 2 (torque=false/true) | 4 (use_base × use_torque) |
| 适用场景 | 固定底盘的双臂操作 | 移动底盘的全身控制 |

## 核心配置参数

```python
# lerobot/src/lerobot/policies/act/configuration_act.py

use_torque: bool = False      # 是否使用力矩信息
use_base: bool = False        # 是否使用底盘速度信息 (NEW)
action_dim_offset: int = 0    # 跳过action前N维
```

## 数据格式

| 字段 | 维度 | 说明 |
|------|------|------|
| action | 17D | [base_vx, base_vy, base_omega] + [left_joints×7] + [right_joints×7] |
| observation.state | 14D | [left_joints×7] + [right_joints×7] |
| observation.effort | 14D | [left_effort×7] + [right_effort×7] |
| observation.base_velocity | 3D | [base_vx, base_vy, base_omega] |

## 训练好的模型

### ACT-wholebody 模型（14D）

| 模型 | HuggingFace | 说明 |
|------|-------------|------|
| torque=false | [B111ue/ACT-100-wholebody-torque-false](https://huggingface.co/B111ue/ACT-100-wholebody-torque-false) | 基线，不使用力矩 |
| torque=true | [B111ue/ACT-100-wholebody-torque-true](https://huggingface.co/B111ue/ACT-100-wholebody-torque-true) | 使用力矩信息 |

### ACT-fullbody 模型（17D）

| 模型 | use_base | use_torque | HuggingFace |
|------|----------|------------|-------------|
| baseline | False | False | [B111ue/ACT-fullbody-baseline](https://huggingface.co/B111ue/ACT-fullbody-baseline) |
| torque-only | False | True | [B111ue/ACT-fullbody-torque-only](https://huggingface.co/B111ue/ACT-fullbody-torque-only) |
| base-only | True | False | [B111ue/ACT-fullbody-base-only](https://huggingface.co/B111ue/ACT-fullbody-base-only) |
| fullbody | True | True | [B111ue/ACT-fullbody-fullbody](https://huggingface.co/B111ue/ACT-fullbody-fullbody) |

## 模型架构

### ACT-wholebody (14D)
```
Transformer Encoder 输入:
[latent(1)] + [state_14d(1)] + [torque_14d(1)] + [image_features(N)]

VAE: action_14d -> latent -> action_14d
```

### ACT-fullbody (17D)
```
Transformer Encoder 输入:
[latent(1)] + [state_17d(1)] + [torque_14d(1)] + [image_features(N)]

VAE: action_17d -> latent -> action_17d
```

## 训练配置

| 参数 | 值 |
|------|-----|
| Steps | 40000 |
| Batch size | 32 |
| GPU | 4卡 (accelerate) |
| Video backend | torchcodec (比pyav快7倍) |
| Save freq | 20000步 |
| Log freq | 100步 |

## 相关资源

- **数据集**: [B111ue/ACT-100-wholebody](https://huggingface.co/datasets/B111ue/ACT-100-wholebody)
- **代码仓库**: [YananZHOU5555/ACT-wholebody-torque](https://github.com/YananZHOU5555/ACT-wholebody-torque)

## License

Apache 2.0
