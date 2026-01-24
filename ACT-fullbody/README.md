# ACT-fullbody

ACT (Action Chunking Transformer) 全身控制版本，支持底盘速度和力矩信息。

## 特性

- **17D Action 输出**：包含底盘速度 (3D) + 双臂关节 (14D)
- **17D State 输入**：底盘速度 (3D) + 双臂关节位置 (14D)
- **14D Torque 输入**：双臂关节力矩
- **4种配置**：支持消融实验

## 模型配置

| 配置名 | use_base | use_torque | State输入 | Torque输入 | HuggingFace |
|--------|----------|------------|-----------|------------|-------------|
| baseline | False | False | [0,0,0] + 14D | zeros(14) | [B111ue/ACT-fullbody-baseline](https://huggingface.co/B111ue/ACT-fullbody-baseline) |
| torque-only | False | True | [0,0,0] + 14D | 14D effort | [B111ue/ACT-fullbody-torque-only](https://huggingface.co/B111ue/ACT-fullbody-torque-only) |
| base-only | True | False | 3D base + 14D | zeros(14) | [B111ue/ACT-fullbody-base-only](https://huggingface.co/B111ue/ACT-fullbody-base-only) |
| fullbody | True | True | 3D base + 14D | 14D effort | [B111ue/ACT-fullbody-fullbody](https://huggingface.co/B111ue/ACT-fullbody-fullbody) |

## 快速开始

### 1. 准备数据

```bash
# 创建数据目录软链接（如果数据已在 ACT-wholebody 中）
cd ACT-fullbody
ln -s ../ACT-wholebody/data data

# 或者下载数据
mkdir -p data
huggingface-cli download B111ue/ACT-100-wholebody --repo-type dataset --local-dir data/ACT-100-wholebody
```

### 2. 训练

```bash
# 训练所有4个配置（串行）
bash train.sh

# 训练完成后会自动生成 loss_comparison.png
```

### 3. 查看结果

训练完成后：
- 模型保存在 `checkpoints/ACT-fullbody-{config}/`
- Loss对比图：`loss_comparison.png`
- 模型自动上传到 HuggingFace

## 训练参数

| 参数 | 值 |
|------|-----|
| Steps | 40000 |
| Batch size | 32 |
| GPU | 4卡 (accelerate) |
| Log freq | 100步 |
| Save freq | 20000步 |
| Video backend | torchcodec |

## 数据格式

| 特征 | 维度 | 说明 |
|------|------|------|
| action | 17D | [base_vx, base_vy, base_omega, left_joints×7, right_joints×7] |
| observation.state | 14D | [left_joints×7, right_joints×7] |
| observation.effort | 14D | [left_effort×7, right_effort×7] |
| observation.base_velocity | 3D | [base_vx, base_vy, base_omega] |

## 模型架构

```
Transformer Encoder 输入序列:
[latent(1)] + [state_17d_embed(1)] + [torque_14d_embed(1)] + [image_features(N)]

VAE:
- Encoder: action_17d -> latent
- Decoder: latent -> action_17d
```

## 与 ACT-wholebody 的区别

| 特性 | ACT-wholebody | ACT-fullbody |
|------|---------------|--------------|
| Action维度 | 14D (只有手臂) | 17D (底盘+手臂) |
| State维度 | 14D | 17D |
| action_dim_offset | 3 (跳过base) | 0 (使用完整action) |
| use_base参数 | 无 | 有 |

## 文件结构

```
ACT-fullbody/
├── train.sh                    # 训练脚本（4个配置串行）
├── plot_loss_comparison.py     # Loss对比图生成
├── README.md                   # 本文档
├── data/                       # 数据集
│   └── ACT-100-wholebody/
└── checkpoints/                # 训练输出
    ├── ACT-fullbody-baseline/
    ├── ACT-fullbody-torque-only/
    ├── ACT-fullbody-base-only/
    └── ACT-fullbody-fullbody/
```

## 相关资源

- **数据集**: [B111ue/ACT-100-wholebody](https://huggingface.co/datasets/B111ue/ACT-100-wholebody)
- **代码仓库**: [YananZHOU5555/ACT-wholebody-torque](https://github.com/YananZHOU5555/ACT-wholebody-torque)
