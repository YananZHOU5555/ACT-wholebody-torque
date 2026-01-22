#!/usr/bin/env python3
"""
ACT-wholebody Policy Deployment for Piper Dual-Arm Robot
支持4种模型配置:
  - torqueFalse_mixFalse: 不使用力矩，不使用base速度
  - torqueFalse_mixTrue:  不使用力矩，使用base速度
  - torqueTrue_mixFalse:  使用力矩，不使用base速度
  - torqueTrue_mixTrue:   使用力矩和base速度

用法:
  python pipier_act_wholebody_v3.py --model torqueTrue_mixTrue
  python pipier_act_wholebody_v3.py --model torqueFalse_mixFalse
"""
from pathlib import Path
import argparse
import json

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
import cv2
import torch
import torch.nn as nn
import pickle


# ==================== Model Definition (from train_wholebody.py) ====================

def get_sinusoid_encoding_table(n_position: int, d_hid: int):
    """Generate sinusoidal positional encoding."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTWholeBodyConfig:
    """Configuration for ACT-wholebody model."""
    def __init__(
        self,
        use_torque: bool = True,
        mix: bool = False,
        state_dim: int = 17,
        action_dim: int = 17,
        chunk_size: int = 100,
        n_obs_steps: int = 1,
        vision_backbone: str = "resnet18",
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_vae_encoder_layers: int = 4,
        dropout: float = 0.1,
        use_vae: bool = True,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        self.use_torque = use_torque
        self.mix = mix
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.vision_backbone = vision_backbone
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_vae_encoder_layers = n_vae_encoder_layers
        self.dropout = dropout
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay


class ACTWholeBodyPolicy(nn.Module):
    """ACT-wholebody Policy with VAE."""

    def __init__(self, config: ACTWholeBodyConfig):
        super().__init__()
        self.config = config

        # Vision backbone
        from torchvision import models
        if config.vision_backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone_out_dim = 512
        else:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_out_dim = 2048
        self.backbone.fc = nn.Identity()

        # Image projection
        self.img_proj = nn.Linear(backbone_out_dim, config.dim_model)

        # State/action projections
        self.state_proj = nn.Linear(config.state_dim, config.dim_model)
        self.torque_proj = nn.Linear(config.state_dim, config.dim_model)
        self.action_proj = nn.Linear(config.action_dim, config.dim_model)

        # VAE encoder
        if config.use_vae:
            self.vae_cls_token = nn.Parameter(torch.randn(1, 1, config.dim_model))
            vae_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.vae_encoder = nn.TransformerEncoder(vae_encoder_layer, num_layers=config.n_vae_encoder_layers)
            self.latent_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

        # Latent to embedding
        self.latent_out_proj = nn.Linear(config.latent_dim, config.dim_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        # Query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, config.chunk_size, config.dim_model))

        # Action head
        self.action_head = nn.Linear(config.dim_model, config.action_dim)

        # Positional encoding
        self.register_buffer('pos_table', get_sinusoid_encoding_table(config.chunk_size + 10, config.dim_model))

    def encode_images(self, images):
        """Encode images using backbone."""
        batch_size = images[0].shape[0]
        all_features = []

        for img in images:
            if img.dim() == 5:
                B, N, C, H, W = img.shape
                img = img.view(B * N, C, H, W)
                feat = self.backbone(img)
                feat = feat.view(B, N, -1)
            else:
                feat = self.backbone(img).unsqueeze(1)
            all_features.append(feat)

        features = torch.cat(all_features, dim=1)
        features = self.img_proj(features)
        return features

    def forward(self, batch, actions=None):
        """Forward pass for training."""
        device = next(self.parameters()).device

        state = batch['observation.state'].to(device)
        batch_size = state.shape[0]

        base_vel = batch.get('observation.base_velocity', torch.zeros(batch_size, 3, device=device))
        if isinstance(base_vel, torch.Tensor):
            base_vel = base_vel.to(device)

        torque = batch.get('observation.effort', torch.zeros(batch_size, 14, device=device))
        if isinstance(torque, torch.Tensor):
            torque = torque.to(device)

        if self.config.mix:
            state_17 = torch.cat([base_vel, state], dim=-1)
        else:
            state_17 = torch.cat([torch.zeros(batch_size, 3, device=device), state], dim=-1)

        if self.config.use_torque:
            torque_17 = torch.cat([torch.zeros(batch_size, 3, device=device), torque], dim=-1)
        else:
            torque_17 = torch.zeros(batch_size, 17, device=device)

        images = []
        for key in ['observation.images.main', 'observation.images.secondary_0',
                    'observation.images.secondary_1', 'observation.images.secondary_2']:
            if key in batch:
                img = batch[key].to(device)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)

        if images:
            img_features = self.encode_images(images)
        else:
            img_features = torch.zeros(batch_size, 1, self.config.dim_model, device=device)

        state_embed = self.state_proj(state_17).unsqueeze(1)
        torque_embed = self.torque_proj(torque_17).unsqueeze(1)

        mu, logvar = None, None
        if self.config.use_vae and actions is not None and False:
            actions = actions.to(device)
            action_embed = self.action_proj(actions)
            cls_token = self.vae_cls_token.expand(batch_size, -1, -1)
            vae_input = torch.cat([cls_token, state_embed, torque_embed, action_embed], dim=1)
            vae_output = self.vae_encoder(vae_input)
            cls_output = vae_output[:, 0]
            latent_params = self.latent_proj(cls_output)
            mu = latent_params[:, :self.config.latent_dim]
            logvar = latent_params[:, self.config.latent_dim:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
        else:
            # TODO
            latent = torch.zeros(batch_size, self.config.latent_dim, device=device)

        latent_embed = self.latent_out_proj(latent).unsqueeze(1)
        encoder_input = torch.cat([latent_embed, state_embed, torque_embed, img_features], dim=1)
        encoder_output = self.encoder(encoder_input)

        queries = self.query_embed.expand(batch_size, -1, -1)
        decoder_output = self.decoder(queries, encoder_output)

        pred_actions = self.action_head(decoder_output)
        return pred_actions, (mu, logvar)

    @torch.no_grad()
    def select_action(self, obs_dict):
        """Inference: select action from observation."""
        device = next(self.parameters()).device
        batch_size = 1

        # Build batch from obs_dict
        batch = {}

        # State (14D arm joints)
        if 'observation.state' in obs_dict:
            state = obs_dict['observation.state']
            if state.dim() == 1:
                state = state.unsqueeze(0)
            batch['observation.state'] = state.to(device)

        # Base velocity (3D) - only used if mix=True
        if 'observation.base_velocity' in obs_dict:
            base_vel = obs_dict['observation.base_velocity']
            if base_vel.dim() == 1:
                base_vel = base_vel.unsqueeze(0)
            batch['observation.base_velocity'] = base_vel.to(device)

        # Torque/effort (14D) - only used if use_torque=True
        if 'observation.effort' in obs_dict:
            effort = obs_dict['observation.effort']
            if effort.dim() == 1:
                effort = effort.unsqueeze(0)
            batch['observation.effort'] = effort.to(device)

        # Images
        for key in ['observation.images.main', 'observation.images.secondary_0',
                    'observation.images.secondary_1', 'observation.images.secondary_2']:
            if key in obs_dict:
                img = obs_dict[key].to(device)
                batch[key] = img

        # Forward pass (inference mode, no actions)
        pred_actions, _ = self.forward(batch, actions=None)

        # Return first action in chunk (17D: 3 base + 14 arm)
        return pred_actions[0, 0, :]  # (17,)


# ==================== Image Normalization (ImageNet) ====================
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==================== Global Buffers ====================
latest_imgs = {"main": None, "wrist_l": None, "wrist_r": None}
latest_q = {"left": None, "right": None}
latest_effort = {"left": None, "right": None}
latest_base_vel = None
smoothed_action = {"left": None, "right": None}


# ==================== ROS Callbacks ====================
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr

def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)

def cb_wrist_l(msg: CompressedImage):
    latest_imgs["wrist_l"] = decode_compressed_image(msg)

def cb_wrist_r(msg: CompressedImage):
    latest_imgs["wrist_r"] = decode_compressed_image(msg)

def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["left"] = np.array(msg.effort, dtype=np.float32)

def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["right"] = np.array(msg.effort, dtype=np.float32)

def cb_base_vel(msg: Odometry):
    """Callback for base odometry - extract velocity from odom message."""
    global latest_base_vel
    latest_base_vel = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z
    ], dtype=np.float32)


# ==================== Preprocessing ====================
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 to normalized float32 tensor (1,3,224,224)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img)


# ==================== Model Loading ====================
def load_policy(model_name: str, checkpoint_dir: str, device: str):
    """Load ACT-wholebody policy from checkpoint with automatic normalization stats."""

    # Parse model config
    use_torque = "torqueTrue" in model_name
    mix = "mixTrue" in model_name

    rospy.loginfo(f"Loading model: {model_name}")
    rospy.loginfo(f"  use_torque: {use_torque}")
    rospy.loginfo(f"  mix: {mix}")

    # Create config
    config = ACTWholeBodyConfig(
        use_torque=use_torque,
        mix=mix,
        state_dim=17,
        action_dim=17,
        chunk_size=100
    )

    # Create model
    model = ACTWholeBodyPolicy(config)

    # Load checkpoint
    ckpt_path = Path(checkpoint_dir) / model_name / "checkpoint_80000.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    rospy.loginfo(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove _orig_mod. prefix from keys (added by torch.compile during training)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Load cleaned state dict
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    rospy.loginfo("Model loaded successfully!")

    # Load normalization stats
    norm_stats = None
    stats_path = Path(checkpoint_dir) / model_name / "norm_stats.pkl"

    if stats_path.exists():
        rospy.loginfo(f"Loading normalization stats from: {stats_path}")
        with open(stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        rospy.loginfo("  ✓ Normalization stats loaded")
    else:
        rospy.logwarn(f"Normalization stats not found at: {stats_path}")
        rospy.logwarn("  Using identity normalization (no scaling)")
        # Create identity normalization stats
        norm_stats = {
            'state_mean': np.zeros(14, dtype=np.float32),
            'state_std': np.ones(14, dtype=np.float32),
            'action_mean': np.zeros(17, dtype=np.float32),
            'action_std': np.ones(17, dtype=np.float32),
        }
        if use_torque:
            norm_stats['effort_mean'] = np.zeros(14, dtype=np.float32)
            norm_stats['effort_std'] = np.ones(14, dtype=np.float32)
        if mix:
            norm_stats['base_vel_mean'] = np.zeros(3, dtype=np.float32)
            norm_stats['base_vel_std'] = np.ones(3, dtype=np.float32)

    return model, config, norm_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["torqueFalse_mixFalse", "torqueFalse_mixTrue",
                                 "torqueTrue_mixFalse", "torqueTrue_mixTrue"],
                        help="Model variant to use")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/zeno/ACT-wholebody/ACT-wholebody/checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--rate", type=int, default=10, help="Control frequency (Hz)")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha")
    args = parser.parse_args()

    rospy.init_node("piper_act_wholebody")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")

    # Load model
    policy, config, norm_stats = load_policy(args.model, args.checkpoint_dir, device)

    # ==================== ROS Setup ====================
    # Image subscribers
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_wrist_l, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_wrist_r, queue_size=1)

    # Joint state subscribers (with effort)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # Base velocity subscriber (only needed if mix=True)
    if config.mix:
        rospy.Subscriber("/ranger_base_node/odom", Odometry, cb_base_vel, queue_size=1)
        rospy.loginfo("Subscribed to base odometry (mix=True)")

    # Publishers
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    rate = rospy.Rate(args.rate)

    rospy.loginfo("=" * 70)
    rospy.loginfo(f"[CONFIG] Model: {args.model}")
    rospy.loginfo(f"  use_torque: {config.use_torque}")
    rospy.loginfo(f"  mix: {config.mix}")
    rospy.loginfo(f"  Control rate: {args.rate} Hz")
    rospy.loginfo(f"  EMA smoothing: alpha={args.smoothing}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0

    while not rospy.is_shutdown():
        # Check required data
        required_ready = (
            all(v is not None for v in latest_imgs.values()) and
            all(v is not None for v in latest_q.values())
        )

        # Additional checks based on config
        if config.mix and latest_base_vel is None:
            rate.sleep()
            continue

        if not required_ready:
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("All sensors ready!")
            data_ready_logged = True

        # ==================== Build Observation ====================
        # Images
        main_img = preprocess_image(latest_imgs["main"]).to(device)
        wrist_l = preprocess_image(latest_imgs["wrist_l"]).to(device)
        wrist_r = preprocess_image(latest_imgs["wrist_r"]).to(device)
        secondary_2 = main_img  # Reuse top camera

        # State: 14D arm joints (left 7 + right 7)
        state_14 = np.concatenate([latest_q["left"][:7], latest_q["right"][:7]], axis=0)

        # Apply normalization to state
        state_14_norm = (state_14 - norm_stats['state_mean']) / norm_stats['state_std']
        state_tensor = torch.from_numpy(state_14_norm).float()

        # Base velocity: 3D (only if mix=True)
        if config.mix and latest_base_vel is not None:
            # Apply normalization to base velocity
            base_vel_norm = (latest_base_vel - norm_stats.get('base_vel_mean', np.zeros(3))) / \
                            norm_stats.get('base_vel_std', np.ones(3))
            base_vel_tensor = torch.from_numpy(base_vel_norm).float()
        else:
            base_vel_tensor = torch.zeros(3)

        # Effort/torque: 14D (only if use_torque=True)
        if config.use_torque:
            effort_left = latest_effort.get("left", np.zeros(7, dtype=np.float32))
            effort_right = latest_effort.get("right", np.zeros(7, dtype=np.float32))
            if effort_left is None:
                effort_left = np.zeros(7, dtype=np.float32)
            if effort_right is None:
                effort_right = np.zeros(7, dtype=np.float32)
            effort_14 = np.concatenate([effort_left[:7], effort_right[:7]], axis=0)
            # Apply normalization to effort
            effort_14_norm = (effort_14 - norm_stats.get('effort_mean', np.zeros(14))) / \
                             norm_stats.get('effort_std', np.ones(14))
            effort_tensor = torch.from_numpy(effort_14_norm).float()
        else:
            effort_tensor = torch.zeros(14)

        obs = {
            "observation.images.main": main_img,
            "observation.images.secondary_0": wrist_l,
            "observation.images.secondary_1": wrist_r,
            "observation.images.secondary_2": secondary_2,
            "observation.state": state_tensor,
            "observation.base_velocity": base_vel_tensor,
            "observation.effort": effort_tensor,
        }

        # ==================== Policy Inference ====================
        with torch.no_grad():
            action_tensor = policy.select_action(obs)

        action_norm = action_tensor.cpu().numpy()  # (17,): [base_vel(3), arm_joints(14)]

        # Apply denormalization to action
        action = action_norm * norm_stats['action_std'] + norm_stats['action_mean']

        # Extract arm actions (skip first 3 base velocity dims)
        action_arms = action[3:]  # (14,)
        action_left = action_arms[:7].copy()
        action_right = action_arms[7:14].copy()

        # ==================== Diagnostic Logging ====================
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count}")
            rospy.loginfo(f"  Input state (14D): {np.array2string(state_14, precision=3)}")
            if config.mix:
                rospy.loginfo(f"  Base velocity: {np.array2string(latest_base_vel, precision=3)}")
            if config.use_torque:
                rospy.loginfo(f"  Effort: {np.array2string(effort_14[:7], precision=3)}...")
            rospy.loginfo(f"  Raw output (17D): {np.array2string(action, precision=3)}")
            rospy.loginfo(f"  Action LEFT (7D):  {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Action RIGHT (7D): {np.array2string(action_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # ==================== EMA Smoothing ====================
        global smoothed_action
        alpha = args.smoothing
        if smoothed_action["left"] is None:
            smoothed_action["left"] = action_left
            smoothed_action["right"] = action_right
        else:
            smoothed_action["left"] = alpha * action_left + (1 - alpha) * smoothed_action["left"]
            smoothed_action["right"] = alpha * action_right + (1 - alpha) * smoothed_action["right"]

        action_left = smoothed_action["left"]
        action_right = smoothed_action["right"]

        # ==================== Publish ====================
        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.position = action_left.tolist()

        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.position = action_right.tolist()

        pub_left.publish(msg_left)
        pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, f"Actions sent (model: {args.model})")

        rate.sleep()


if __name__ == "__main__":
    main()
