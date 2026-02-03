#!/usr/bin/env python3
"""
ACT Policy Deployment for Piper Dual-Arm Robot with Mobile Base (LeRobot)
Model: ACT-final-promax (fullbody: use_base=true, use_torque=true)

Key features:
1. State input normalization (14D: left_arm 7D + right_arm 7D)
2. Effort/Torque input normalization (14D: left_arm 7D + right_arm 7D)
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


# ====== Normalization statistics (extracted from ACT-final-V30 dataset) ======

# State normalization parameters (observation.state) - 14D
STATE_MEAN = np.array([
    -0.08764796117262724, 1.9186290527808452, -1.4674627811498206, -0.14788388498849783,
    0.8725728076236958, 0.02271875563396174, 0.030379420150038364,
    0.13536423722593507, 1.9179583322532239, -1.523341596600525, -0.020322882023583594,
    0.8603840853183115, -0.056340599302742746, 0.04087620689273601
], dtype=np.float32)

STATE_STD = np.array([
    0.1389286170949437, 0.656068269899939, 0.44292394141196123, 0.3011435751198227,
    0.3152024513695464, 0.2345525242018114, 0.023870717352886265,
    0.1927205550244276, 0.6723317365751659, 0.522974646537298, 0.1507983091653653,
    0.27794210210420506, 0.25908047459724953, 0.02675731087601505
], dtype=np.float32)

# Velocity normalization parameters (observation.velocity) - 14D
VELOCITY_MEAN = np.array([
    -0.0001922637327392761, -5.7321190363626374e-05, 1.1903865542210835e-07, -0.0001654970565949593,
    1.3863956359684968e-05, -0.00014588305902721936, 0.0,
    0.00020563026393388912, 0.00017323577332001498, -0.00012519665447014987, -2.122103948947641e-05,
    0.00016270892662398486, 0.00014612398231672434, 0.0
], dtype=np.float32)

VELOCITY_STD = np.array([
    0.09826098395643831, 0.2730686241471781, 0.16866774586450986, 0.17809314466310588,
    0.19778873729240187, 0.12254881021549655, 1e-6,
    0.09353596613273667, 0.2597470514492083, 0.20095663388952464, 0.08683953914104689,
    0.16815808218483025, 0.10192764097389737, 1e-6
], dtype=np.float32)

# Effort/Torque normalization parameters (observation.effort) - 14D
EFFORT_MEAN = np.array([
    -0.053246605184523725, -1.9588883270603827, -1.0818609839630418, -0.05039749202038619,
    -0.3861511290438718, 0.11525426951139985, -0.9619375799105806,
    0.1370144918679899, -1.8916983574363322, -1.0579076804363785, 0.07381057493137111,
    -0.4103718386739903, -0.09851880311850342, -0.8807574816562687
], dtype=np.float32)

EFFORT_STD = np.array([
    0.24030533192352757, 0.8724248423724753, 0.2764448465015082, 0.3328059891946098,
    0.46834984314202016, 0.29242375454042235, 0.9715699955971502,
    0.28624747947260637, 0.8870365279328892, 0.25323945052344893, 0.4261827892102537,
    0.564531732107735, 0.4329945838289876, 1.016950860568518
], dtype=np.float32)

# Base Velocity normalization parameters (observation.base_velocity) - 3D
BASE_VELOCITY_MEAN = np.array([
    0.02475262523178208, 0.00034517253897649867, 0.1490651561459677
], dtype=np.float32)

BASE_VELOCITY_STD = np.array([
    0.09049152585859885, 0.009364309364439195, 0.44805979763268977
], dtype=np.float32)

# Action denormalization parameters - 17D (first 3D is base velocity, last 14D is arm joints)
ACTION_MEAN = np.array([
    0.02475262523178208, 0.00034517253897649867, 0.1490651561459677,  # base velocity
    -0.08783563584127986, 1.8841609827989585, -1.495462734479947, -0.15382803226622613,
    0.8337939932394746, 0.03172244912584612, 0.022063803121682844,  # left arm
    0.13920889206745177, 1.8819337069516997, -1.5435024215919273, -0.015018790526428807,
    0.8281660511634434, -0.06655796048728584, 0.03682672354387212  # right arm
], dtype=np.float32)

ACTION_STD = np.array([
    0.09049152585859885, 0.009364309364439195, 0.44805979763268977,  # base velocity
    0.14045296118288658, 0.6430018097339159, 0.44412451301551314, 0.3202675573527283,
    0.3339424496318185, 0.23328308435311662, 0.04712699824983289,  # left arm
    0.1938751101578785, 0.6582704827185913, 0.5241329063349904, 0.15954044613695628,
    0.2991715816752588, 0.25277604636088585, 0.034500066532733686  # right arm
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


# ====== Default model path ======
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "ACT-fold-clothes" / "checkpoints" / "020000" / "pretrained_model"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ACT-final-promax Policy Deployment")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"Checkpoint path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA smoothing alpha (0=no smoothing, 1=no history)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable EMA smoothing")

    args, _ = parser.parse_known_args()

    ckpt_dir = args.ckpt

    # Initialize ROS node
    rospy.init_node("piper_act_final_promax")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")
    rospy.loginfo(f"Model path: {ckpt_dir}")

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
    rospy.loginfo(f"  Model: ACT-final-promax")
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
        # Logic: when omega(z) is 0, forward mode (x=vx, y=0)
        #        when omega(z) is not 0, lateral mode (x=0, y=vx)
        cmd_vel = Twist()
        if use_base:
            if abs(action_base[2]) < 0.05:  # omega close to 0, forward mode
                cmd_vel.linear.x = float(action_base[0])
                cmd_vel.linear.y = float(action_base[1])
                cmd_vel.angular.z = float(action_base[2])
            elif abs(action_base[2]) >= 0.1:  # omega close to 0, forward mode
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = float(action_base[0])
                cmd_vel.angular.z = 0.0
            else:  # omega not 0, lateral mode
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = float(action_base[0])
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
