#!/usr/bin/env python3
"""
ACT Policy Deployment for Piper Dual-Arm Robot with Mobile Base (LeRobot)
Model: hanger-wholebody models (trained on ACT-100-WHOLE-V30 dataset)

Supports four models (select via --1, --2, --3, --4):
- --1: ACT-fullbody-arm-only    (no base velocity, no torque)
- --2: ACT-fullbody-arm-torque  (no base velocity, with torque)
- --3: ACT-fullbody-base-only   (with base velocity, no torque)
- --4: ACT-fullbody-fullbody    (with base velocity + torque)

Key features:
1. State input normalization (14D: left_arm 7D + right_arm 7D)
2. Effort/Torque input normalization (14D: left_arm 7D + right_arm 7D) - only for torque models
3. Base Velocity input normalization (3D: vx, vy, omega)
4. Image input ImageNet normalization (224x224, 4 cameras)
5. Action output denormalization (17D: base 3D + left_arm 7D + right_arm 7D)
6. Control frequency matches dataset fps (10Hz)
"""
import argparse
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


# ====== Normalization statistics (extracted from ACT-100-WHOLE-V30 dataset) ======

# State normalization parameters (observation.state) - 14D
STATE_MEAN = np.array([
    -0.13800235, 0.708229, -0.76164, -0.46664125, 0.9283637, 0.69568306, 0.03746549,
    -0.06074022, 0.73452294, -0.7104236, 0.10306202, 0.89564264, -0.36123428, 0.0519936
], dtype=np.float32)

STATE_STD = np.array([
    0.09885322, 0.6163445, 0.36601868, 0.6004334, 0.29522428, 0.92954695, 0.02679501,
    0.6185341, 0.805953, 0.34201962, 0.21602891, 0.20932098, 0.500204, 0.02783796
], dtype=np.float32)

# Velocity normalization parameters (observation.velocity) - 14D
VELOCITY_MEAN = np.array([
    -4.7408353e-04, -8.8127439e-05, 4.2396019e-05, 2.6840469e-04,
    -5.8893293e-05, -1.8512804e-04, 0.0,
    -6.9834314e-05, -4.0171304e-05, 1.8326264e-06, 1.8801710e-04,
    -5.5529490e-05, -1.4769826e-04, 0.0
], dtype=np.float32)

VELOCITY_STD = np.array([
    0.06905802, 0.16672567, 0.12698808, 0.19476354, 0.13011919, 0.27415988, 1e-6,
    0.20765626, 0.22588256, 0.12734707, 0.11301319, 0.12797238, 0.18495426, 1e-6
], dtype=np.float32)

# Effort/Torque normalization parameters (observation.effort) - 14D
EFFORT_MEAN = np.array([
    -0.01386216, -0.36312038, -1.0782875, -0.38379958, -0.6008535, 0.03243913, -0.6881537,
    0.0290641, -0.4250933, -1.0590883, 0.11176357, -0.74314135, 0.02537749, -0.30430987
], dtype=np.float32)

EFFORT_STD = np.array([
    0.14264308, 0.5510519, 0.195174, 0.5067644, 0.8426294, 0.15906096, 0.9409513,
    0.11192082, 0.8166702, 0.14112343, 0.21373712, 0.4619316, 0.10010264, 0.6505068
], dtype=np.float32)

# Base Velocity normalization parameters (observation.base_velocity) - 3D
BASE_VELOCITY_MEAN = np.array([
    -0.01346323, 0.0, 0.0
], dtype=np.float32)

BASE_VELOCITY_STD = np.array([
    0.04010911, 1e-6, 1e-6  # vy and omega std=0, use 1e-6 to avoid division by zero
], dtype=np.float32)

# Action denormalization parameters - 17D (first 3D is base velocity, last 14D is arm joints)
ACTION_MEAN = np.array([
    -0.01346323, 0.0, 0.0,  # base velocity (fixed: x has actual motion, y/omega=0)
    -0.13756745, 0.7017483, -0.7814063, -0.5027233, 0.8788727, 0.697639, 0.03894639,  # left arm
    -0.05901453, 0.72729087, -0.72974867, 0.11184663, 0.83339465, -0.36014965, 0.05324022  # right arm
], dtype=np.float32)

ACTION_STD = np.array([
    0.04010911, 1e-6, 1e-6,  # base velocity (fixed: x has actual motion, y/omega std=0, use 1e-6 to avoid division by zero)
    9.9508233e-02, 6.0909957e-01, 3.6958531e-01, 6.4608008e-01, 3.5777524e-01, 9.3535048e-01, 4.9930137e-02,  # left arm
    6.2019479e-01, 7.9297173e-01, 3.4270123e-01, 2.3351796e-01, 2.2375561e-01, 5.0174183e-01, 3.2039780e-02  # right arm
], dtype=np.float32)

# Image normalization parameters (ImageNet standard)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ====== Global data cache ======
latest_imgs = {
    "main": None,
    "secondary_0": None,  # wrist_left
    "secondary_1": None,  # wrist_right
    "secondary_2": None,  # reuse main
}

latest_q = {
    "left": None,
    "right": None,
}

latest_effort = {
    "left": None,
    "right": None,
}

latest_velocity = {
    "left": None,
    "right": None,
}

latest_base_velocity = None

smoothed_action = {
    "left": None,
    "right": None,
    "base": None,
}


# ====== Helper functions ======
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """Decode compressed image message to BGR numpy array"""
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


# ====== ROS callback functions ======
def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)

def cb_secondary_0(msg: CompressedImage):
    latest_imgs["secondary_0"] = decode_compressed_image(msg)

def cb_secondary_1(msg: CompressedImage):
    latest_imgs["secondary_1"] = decode_compressed_image(msg)

def cb_secondary_2(msg: CompressedImage):
    latest_imgs["secondary_2"] = decode_compressed_image(msg)


def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["left"] = np.array(msg.effort, dtype=np.float32)
    if msg.velocity:
        latest_velocity["left"] = np.array(msg.velocity, dtype=np.float32)

def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["right"] = np.array(msg.effort, dtype=np.float32)
    if msg.velocity:
        latest_velocity["right"] = np.array(msg.velocity, dtype=np.float32)


def cb_odom(msg: Odometry):
    """Get base velocity from odometry"""
    global latest_base_velocity
    latest_base_velocity = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z
    ], dtype=np.float32)


# ====== Preprocessing functions ======
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 (H,W,3) to normalized float32 torch tensor (1,3,224,224)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img)


def normalize_state(state: np.ndarray) -> np.ndarray:
    return (state - STATE_MEAN) / STATE_STD


def normalize_velocity(velocity: np.ndarray) -> np.ndarray:
    return (velocity - VELOCITY_MEAN) / VELOCITY_STD


def normalize_effort(effort: np.ndarray) -> np.ndarray:
    return (effort - EFFORT_MEAN) / EFFORT_STD


def normalize_base_velocity(base_vel: np.ndarray) -> np.ndarray:
    return (base_vel - BASE_VELOCITY_MEAN) / BASE_VELOCITY_STD


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Unnormalize 17D action output."""
    return action * ACTION_STD + ACTION_MEAN


# ====== Model loading ======
def load_policy(ckpt_dir: str, device: str) -> ACTPolicy:
    """Load ACT policy model"""
    local_path = Path(ckpt_dir).expanduser().resolve()
    if local_path.exists():
        pretrained_path = str(local_path)
        rospy.loginfo(f"Loading ACT Policy from local path: {pretrained_path}")
    else:
        pretrained_path = ckpt_dir
        rospy.loginfo(f"Loading ACT Policy from Hugging Face: {pretrained_path}")

    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=pretrained_path)
    policy = policy.to(device)

    rospy.loginfo("=" * 70)
    rospy.loginfo("[INFO] Policy loaded successfully")
    rospy.loginfo(f"  use_torque: {policy.config.use_torque}")
    rospy.loginfo(f"  use_base: {policy.config.use_base}")
    rospy.loginfo(f"  action_dim_offset: {policy.config.action_dim_offset}")
    rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
    rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
    rospy.loginfo("=" * 70)

    policy.eval()
    return policy


# ====== Model configuration ======
# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_CONFIGS = {
    "1": {
        "name": "arm-only",
        "path": "ACT-fullbody-arm-only",
        "description": "no base velocity, no torque"
    },
    "2": {
        "name": "arm-torque",
        "path": "ACT-fullbody-arm-torque",
        "description": "no base velocity, with torque"
    },
    "3": {
        "name": "base-only",
        "path": "ACT-fullbody-base-only",
        "description": "with base velocity, no torque"
    },
    "4": {
        "name": "fullbody",
        "path": "ACT-fullbody-fullbody",
        "description": "with base velocity + torque"
    },
}

# Default model path
DEFAULT_MODEL_PATH = MODELS_DIR / "ACT-hanger" / "ACT-fullbody-fullbody"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ACT Hanger-Wholebody Policy Deployment")
    parser.add_argument("--1", dest="model_1", action="store_true",
                        help="Use arm-only model (no base, no torque)")
    parser.add_argument("--2", dest="model_2", action="store_true",
                        help="Use arm-torque model (no base, with torque)")
    parser.add_argument("--3", dest="model_3", action="store_true",
                        help="Use base-only model (with base, no torque)")
    parser.add_argument("--4", dest="model_4", action="store_true",
                        help="Use fullbody model (with base + torque)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Custom checkpoint path (overrides --1/2/3/4)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA smoothing alpha (0=no smoothing, 1=no history)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable EMA smoothing")

    args, _ = parser.parse_known_args()

    # Determine which model to use
    model_key = None
    if args.model_1:
        model_key = "1"
    elif args.model_2:
        model_key = "2"
    elif args.model_3:
        model_key = "3"
    elif args.model_4:
        model_key = "4"

    if args.ckpt:
        ckpt_dir = args.ckpt
        model_name = "custom"
        model_desc = f"Custom model: {args.ckpt}"
    elif model_key:
        config = MODEL_CONFIGS[model_key]
        ckpt_dir = str(MODELS_DIR / "ACT-hanger" / config['path'])
        model_name = config["name"]
        model_desc = config["description"]
    else:
        # Default to fullbody model
        ckpt_dir = str(DEFAULT_MODEL_PATH)
        model_name = "fullbody"
        model_desc = "with base velocity + torque (default)"

    # Initialize ROS node
    rospy.init_node("piper_act_hanger")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")
    rospy.loginfo(f"Model: {model_name} - {model_desc}")

    # Load model
    policy = load_policy(ckpt_dir, device)
    policy.reset()

    use_torque = policy.config.use_torque
    use_base = policy.config.use_base

    # ====== ROS Subscribers ======
    # Image subscriptions
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_secondary_0, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_secondary_1, queue_size=1)
    # secondary_2 reuses main (consistent with data collection)

    # Joint state subscriptions (includes position, velocity, effort)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # Base odometry subscription
    rospy.Subscriber("/ranger_base_node/odom", Odometry, cb_odom, queue_size=1)

    # ====== ROS Publishers ======
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    rate = rospy.Rate(args.rate)

    ENABLE_SMOOTHING = not args.no_smoothing
    SMOOTHING_ALPHA = args.smoothing

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] Deployment settings:")
    rospy.loginfo(f"  Model: {model_name}")
    rospy.loginfo(f"  use_torque: {use_torque}")
    rospy.loginfo(f"  use_base: {use_base}")
    rospy.loginfo(f"  Control rate: {args.rate} Hz")
    rospy.loginfo(f"  EMA smoothing: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0
    global latest_base_velocity

    while not rospy.is_shutdown():
        # Check if required data is ready
        # Images: main, secondary_0, secondary_1 must be available
        if (latest_imgs["main"] is None or
            latest_imgs["secondary_0"] is None or
            latest_imgs["secondary_1"] is None):
            rate.sleep()
            continue

        # Joint states must be available
        if latest_q["left"] is None or latest_q["right"] is None:
            rate.sleep()
            continue

        # If using torque model, effort must be available
        if use_torque and (latest_effort["left"] is None or latest_effort["right"] is None):
            rate.sleep()
            continue

        # If using base model, base velocity must be available
        if use_base and latest_base_velocity is None:
            rate.sleep()
            continue

        # If not using base but no base_velocity data, fill with zeros
        if not use_base and latest_base_velocity is None:
            latest_base_velocity = np.zeros(3, dtype=np.float32)

        if not data_ready_logged:
            rospy.loginfo("All required sensors ready, starting inference...")
            data_ready_logged = True

        # ====== Build observation dictionary ======
        # Image preprocessing
        main_img = preprocess_image(latest_imgs["main"]).to(device)
        secondary_0 = preprocess_image(latest_imgs["secondary_0"]).to(device)
        secondary_1 = preprocess_image(latest_imgs["secondary_1"]).to(device)
        # secondary_2: if no 4th camera, reuse main
        if latest_imgs["secondary_2"] is not None:
            secondary_2 = preprocess_image(latest_imgs["secondary_2"]).to(device)
        else:
            secondary_2 = main_img

        # State normalization (14D)
        state_raw = np.concatenate([latest_q["left"], latest_q["right"]], axis=0).astype(np.float32)
        state_normalized = normalize_state(state_raw)
        state = torch.from_numpy(state_normalized[None, :]).to(device)

        # Velocity normalization (14D)
        if latest_velocity["left"] is not None and latest_velocity["right"] is not None:
            velocity_raw = np.concatenate([latest_velocity["left"], latest_velocity["right"]], axis=0).astype(np.float32)
        else:
            velocity_raw = np.zeros(14, dtype=np.float32)
        velocity_normalized = normalize_velocity(velocity_raw)
        velocity = torch.from_numpy(velocity_normalized[None, :]).to(device)

        # Effort/Torque normalization (14D)
        if latest_effort["left"] is not None and latest_effort["right"] is not None:
            effort_raw = np.concatenate([latest_effort["left"], latest_effort["right"]], axis=0).astype(np.float32)
        else:
            effort_raw = np.zeros(14, dtype=np.float32)
        effort_normalized = normalize_effort(effort_raw)
        effort = torch.from_numpy(effort_normalized[None, :]).to(device)

        # Base Velocity normalization (3D)
        base_vel_raw = latest_base_velocity.copy()
        base_vel_normalized = normalize_base_velocity(base_vel_raw)
        base_velocity = torch.from_numpy(base_vel_normalized[None, :]).to(device)

        # Build observation dictionary
        obs = {
            "observation.images.main": main_img,
            "observation.images.secondary_0": secondary_0,
            "observation.images.secondary_1": secondary_1,
            "observation.images.secondary_2": secondary_2,
            "observation.state": state,
            "observation.velocity": velocity,
            "observation.effort": effort,
            "observation.base_velocity": base_velocity,
        }

        # ====== Model inference ======
        with torch.no_grad():
            action_tensor = policy.select_action(obs)

        if action_tensor.dim() == 2:
            action_normalized = action_tensor[0, :].cpu().numpy()
        else:
            action_normalized = action_tensor.cpu().numpy()

        # Action denormalization (17D)
        action = unnormalize_action(action_normalized)

        if len(action) != 17:
            rospy.logwarn(f"Invalid action dim: {len(action)}, expected 17")
            rate.sleep()
            continue

        # Split action: [base_vx, base_vy, base_omega, left_arm x7, right_arm x7]
        action_base = action[0:3].copy()
        action_left = action[3:10].copy()
        action_right = action[10:17].copy()

        # ====== Diagnostic logs ======
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count}")
            rospy.loginfo(f"  Raw state LEFT:        {np.array2string(latest_q['left'], precision=3)}")
            rospy.loginfo(f"  Raw state RIGHT:       {np.array2string(latest_q['right'], precision=3)}")
            if use_torque:
                rospy.loginfo(f"  Raw effort LEFT:       {np.array2string(latest_effort['left'], precision=3)}")
                rospy.loginfo(f"  Raw effort RIGHT:      {np.array2string(latest_effort['right'], precision=3)}")
            if use_base:
                rospy.loginfo(f"  Raw base velocity:     vx={base_vel_raw[0]:.4f}, vy={base_vel_raw[1]:.4f}, omega={base_vel_raw[2]:.4f}")
            rospy.loginfo(f"  Model output (norm):   {np.array2string(action_normalized[:5], precision=3)}...")
            rospy.loginfo(f"  Unnorm action BASE:    vx={action_base[0]:.4f}, vy={action_base[1]:.4f}, omega={action_base[2]:.4f}")
            rospy.loginfo(f"  Unnorm action LEFT:    {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Unnorm action RIGHT:   {np.array2string(action_right, precision=3)}")

            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]
            rospy.loginfo(f"  Delta LEFT:            {np.array2string(delta_left, precision=3)}")
            rospy.loginfo(f"  Delta RIGHT:           {np.array2string(delta_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # ====== EMA smoothing ======
        global smoothed_action
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
                smoothed_action["base"] = action_base
            else:
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
                smoothed_action["base"] = SMOOTHING_ALPHA * action_base + (1.0 - SMOOTHING_ALPHA) * smoothed_action["base"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]
            action_base = smoothed_action["base"]

        # ====== Publish control commands ======
        # Base velocity command
        cmd_vel = Twist()
        if use_base:
            cmd_vel.linear.x = float(action_base[0])  # Output x directly, no inversion or scaling
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
        pub_cmd_vel.publish(cmd_vel)

        # Left arm joint command
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()
        pub_left.publish(msg_left)

        # Right arm joint command
        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()
        pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, f"Actions sent (torque={use_torque}, base={use_base}, base_vx={action_base[0]:.4f})")

        rate.sleep()


if __name__ == "__main__":
    main()
