# ACT Whole-Body Control: Hyperparameter Summary

This document provides a comprehensive summary of all hyperparameters used in our ACT (Action Chunking Transformer) whole-body control implementation.

## 1. Model Architecture

### 1.1 Vision Backbone (ResNet-18)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Backbone | ResNet-18 | Torchvision ResNet variant for image encoding |
| Pretrained Weights | ImageNet1K_V1 | Pretrained on ImageNet-1K dataset |
| Replace Final Stride with Dilation | False | Whether to use dilated convolution in final layer |
| Input Image Size | 224 × 224 × 3 | RGB image resolution |
| Number of Cameras | 4 | Top, left wrist, right wrist, auxiliary |
| Image Normalization | ImageNet Mean/Std | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |

### 1.2 Transformer Encoder

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden Dimension (d_model) | 512 | Main hidden dimension of transformer |
| Number of Attention Heads | 8 | Multi-head attention heads |
| Feed-Forward Dimension | 3200 | Dimension of FFN intermediate layer |
| Number of Encoder Layers | 4 | Transformer encoder depth |
| Feed-Forward Activation | ReLU | Activation function in FFN |
| Pre-Normalization | False | Layer normalization placement |
| Dropout | 0.1 | Dropout rate in transformer layers |

### 1.3 Transformer Decoder

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden Dimension (d_model) | 512 | Same as encoder |
| Number of Attention Heads | 8 | Multi-head attention heads |
| Feed-Forward Dimension | 3200 | Dimension of FFN intermediate layer |
| Number of Decoder Layers | 1 | Transformer decoder depth |
| Feed-Forward Activation | ReLU | Activation function in FFN |
| Dropout | 0.1 | Dropout rate in transformer layers |

### 1.4 VAE (Variational Autoencoder)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Use VAE | True | Enable variational objective |
| Latent Dimension | 32 | Dimension of latent space |
| Number of VAE Encoder Layers | 4 | VAE encoder transformer depth |
| KL Weight | 10.0 | Weight for KL-divergence loss term |

### 1.5 Action Chunking

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk Size | 100 | Number of action steps predicted per inference |
| Number of Action Steps | 100 | Actions executed per policy call |
| Number of Observation Steps | 1 | Observation history length |
| Temporal Ensemble Coefficient | None | Disabled (optional: 0.01 for ensembling) |

## 2. Input/Output Dimensions

### 2.1 Observation Space

| Feature | Dimension | Description |
|---------|-----------|-------------|
| observation.state | 14D | Joint positions (left arm 7D + right arm 7D) |
| observation.velocity | 14D | Joint velocities (left arm 7D + right arm 7D) |
| observation.effort | 14D | Joint torques (left arm 7D + right arm 7D) |
| observation.base_velocity | 3D | Base velocity (vx, vy, omega) |
| observation.images.* | 224 × 224 × 3 | RGB images from 4 cameras |

### 2.2 Action Space

| Feature | Dimension | Description |
|---------|-----------|-------------|
| action | 17D | Full-body action |
| - Base velocity | 3D | [vx, vy, omega] |
| - Left arm joints | 7D | 7-DOF joint positions |
| - Right arm joints | 7D | 7-DOF joint positions |

### 2.3 Extended State Input (Whole-Body)

| Configuration | State Dimension | Description |
|---------------|-----------------|-------------|
| use_base=False | 17D (3D zeros + 14D arm) | Base velocity padded with zeros |
| use_base=True | 17D (3D base + 14D arm) | Full base velocity included |

## 3. Training Configuration

### 3.1 Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | AdamW | Optimizer type |
| Learning Rate | 1e-5 | Main learning rate |
| Backbone Learning Rate | 1e-5 | Learning rate for ResNet backbone |
| Weight Decay | 1e-4 | L2 regularization coefficient |
| Batch Size | 32 | Training batch size |

### 3.2 Training Schedule

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total Training Steps | 40,000 | Maximum training iterations |
| Save Frequency | 20,000 | Checkpoint saving interval |
| Evaluation Frequency | 20,000 | Evaluation interval |
| Log Frequency | 100 | Logging interval |

### 3.3 Data Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Dataset FPS | 10 Hz | Data collection frequency |
| Control Frequency | 10 Hz | Deployment control rate |
| Video Backend | PyAV | Video decoding backend |
| Normalization Mode | Mean-Std | For state, action, and visual features |

## 4. Extension Parameters

### 4.1 Torque Extension

| Parameter | Value | Description |
|-----------|-------|-------------|
| use_torque | True/False | Enable torque information fusion |
| Torque Input Dimension | 14D | Same as joint state dimension |
| Torque Token | Added to encoder | Additional transformer input token |

### 4.2 Full-Body Extension

| Parameter | Value | Description |
|-----------|-------|-------------|
| use_base | True/False | Enable base velocity control |
| action_dim_offset | 0 | Skip first N action dimensions (0 = use all) |
| Base Velocity Dimension | 3D | [vx, vy, omega] |

## 5. Model Variants

| Variant | use_base | use_torque | Description |
|---------|----------|------------|-------------|
| arm-only | False | False | Dual-arm control only |
| arm-torque | False | True | Dual-arm with torque feedback |
| base-only | True | False | Full-body without torque |
| fullbody | True | True | Full-body with torque feedback |

## 6. Deployment Configuration

### 6.1 Inference Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Control Rate | 10 Hz | Real-time control frequency |
| EMA Smoothing Alpha | 0.3 | Exponential moving average coefficient |
| Device | CUDA | GPU acceleration |

### 6.2 Action Smoothing (Optional)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Enable Smoothing | True | Apply EMA to actions |
| Smoothing Alpha | 0.3 | α=0: no smoothing, α=1: no history |

## 7. Loss Function

The total loss is computed as:

$$\mathcal{L} = \mathcal{L}_{L1} + \lambda_{KL} \cdot \mathcal{L}_{KL}$$

| Component | Formula | Weight |
|-----------|---------|--------|
| L1 Reconstruction Loss | $\mathcal{L}_{L1} = \|a - \hat{a}\|_1$ | 1.0 |
| KL Divergence Loss | $\mathcal{L}_{KL} = D_{KL}(q(z|x) \| p(z))$ | 10.0 |

## 8. Summary Table

| Category | Parameter | Value |
|----------|-----------|-------|
| **Architecture** | | |
| | Vision Backbone | ResNet-18 (ImageNet pretrained) |
| | Transformer d_model | 512 |
| | Attention Heads | 8 |
| | FFN Dimension | 3200 |
| | Encoder Layers | 4 |
| | Decoder Layers | 1 |
| | VAE Latent Dim | 32 |
| | VAE Encoder Layers | 4 |
| **Action Chunking** | | |
| | Chunk Size | 100 |
| | Action Steps | 100 |
| **Training** | | |
| | Optimizer | AdamW |
| | Learning Rate | 1e-5 |
| | Weight Decay | 1e-4 |
| | Batch Size | 32 |
| | Training Steps | 40,000 |
| | KL Weight | 10.0 |
| | Dropout | 0.1 |
| **Data** | | |
| | Image Size | 224 × 224 |
| | Number of Cameras | 4 |
| | State Dimension | 14D (arm) + 3D (base) |
| | Action Dimension | 17D |
| | Control Frequency | 10 Hz |
