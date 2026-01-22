# ACT-wholebody-torque

ACT (Action Chunking Transformer) 策略的 torque 扩展版本，基于 LeRobot 框架。

## 项目结构

```
ACT-wholebody/
├── ACT-wholebody/
│   ├── train.sh          # 训练脚本
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

## 服务器部署

### 1. 克隆仓库
```bash
git clone https://github.com/YananZHOU5555/ACT-wholebody-torque.git
cd ACT-wholebody-torque
```

### 2. 创建环境并安装
```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ./lerobot
```

### 3. 下载数据集
```bash
mkdir -p ACT-wholebody/data
cd ACT-wholebody/data
# 从 HuggingFace 下载
huggingface-cli download B111ue/ACT-100-wholebody --repo-type dataset --local-dir ACT-100-wholebody
```

### 4. 运行训练
```bash
cd ACT-wholebody
./train.sh > train.log 2>&1 &

# 监控训练进度
python monitor.py
```

## 训练配置 (train.sh)

- **步数**: 40000 steps × 2 (torque=false, torque=true)
- **保存频率**: 每20000步
- **Batch size**: 32
- **Action维度**: 14D (跳过前3D base velocity)
- **输出目录**: `checkpoints/ACT-100-wholebody-torque-{false,true}`

## 数据说明

| 字段 | 维度 | 说明 |
|------|------|------|
| observation.state | 14D | 双臂关节位置 |
| observation.effort | 14D | 双臂关节力矩 |
| action | 17D | 3D base + 14D arms (训练时只用后14D) |

## 相关资源

- **数据集**: [B111ue/ACT-100-wholebody](https://huggingface.co/datasets/B111ue/ACT-100-wholebody)
- **代码仓库**: [YananZHOU5555/ACT-wholebody-torque](https://github.com/YananZHOU5555/ACT-wholebody-torque)
