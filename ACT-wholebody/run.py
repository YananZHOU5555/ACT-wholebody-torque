#!/usr/bin/env python3
"""
ACT Policy Deployment for Piper Dual-Arm Robot (LeRobot)
Model: ACT-100-wholebody-torque (with torque extension)

支持两个模型:
- torque=false: B111ue/ACT-100-wholebody-torque-false (基线模型)
- torque=true:  B111ue/ACT-100-wholebody-torque-true  (使用力矩信息)

关键特性：
1. State 输入归一化 (14D: 左臂7D + 右臂7D)
2. Effort/Torque 输入归一化 (14D: 左臂7D + 右臂7D) - 仅torque=true模型使用
3. Image 输入 ImageNet 归一化 (224x224, 4个相机)
4. Action 输出反归一化 (17D -> 14D, 跳过前3D base velocity)
5. 控制频率匹配数据集 fps
"""
import argparse
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped, Twist
import cv2
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


# ====== 归一化统计量（从 ACT-100-wholebody 数据集提取）======

# State 归一化参数 (observation.state) - 14D
STATE_MEAN = np.array([
    -0.2606253817248191, 0.7552818851951077, -0.902833292585358, 0.027076443363946816,
    0.8637454917548779, -0.12004155569818943, 0.05240164214775147,
    0.09250259225547974, 0.8275684382943502, -0.949230392663978, -0.0421798839430777,
    0.8844605928899448, -0.06516013183879305, 0.06623479914021548
], dtype=np.float32)

STATE_STD = np.array([
    0.1458120545222341, 1.0039213821238118, 0.6252098002135726, 0.15901794525542204,
    0.15923122950471855, 0.15031394787355784, 0.016539987686139,
    0.2155741393817632, 1.0577767468708403, 0.7097075497516163, 0.0751885513953223,
    0.12686351612490318, 0.16542830176214737, 0.007889084559677658
], dtype=np.float32)

# Effort/Torque 归一化参数 (observation.effort) - 14D
EFFORT_MEAN = np.array([
    0.05346175895726731, -0.5258397613112038, -1.0951317199946384, 0.038133785473922696,
    -0.9552784890214489, -0.012632157541295035, -0.2073359791634767,
    -0.04887072005540305, -0.605369531870819, -1.1127463392558585, -0.004627995089386135,
    -0.9290238148816207, 0.041828908540318094, -0.26164368069322597
], dtype=np.float32)

EFFORT_STD = np.array([
    0.13921318838314992, 1.327231869605766, 0.18275076012801156, 0.164594201426091,
    0.30531892590097426, 0.07180175091251256, 0.6783889318493697,
    0.11910206822304907, 1.3984141042097986, 0.14573296778535322, 0.11277661514642591,
    0.3626070647621022, 0.15866254365823784, 0.7505182660194688
], dtype=np.float32)

# Velocity 归一化参数 (observation.velocity) - 14D
VELOCITY_MEAN = np.array([
    -0.0014445387484818557, 6.286478971171174e-05, -3.143180000978738e-05, 0.0008840662253880622,
    -1.8355178156941957e-05, -0.0010493440953454206, 0.0,
    0.0008203886692106279, -7.22782418334331e-05, 6.465994366763633e-05, -0.00051808828784815,
    -7.535701693278058e-05, 0.0005321255898518633, 0.0
], dtype=np.float32)

VELOCITY_STD = np.array([
    0.05932093548518854, 0.2792191180156555, 0.21266281120346542, 0.07863654070870085,
    0.11849771180740613, 0.0826882753653537, 1e-6,  # 避免除零
    0.1031496435088229, 0.2992049627675998, 0.2170811225548048, 0.0501174043301144,
    0.10663618064167435, 0.06397054704659563, 1e-6  # 避免除零
], dtype=np.float32)

# EE Pose 归一化参数 (observation.ee_pose) - 14D
EE_POSE_MEAN = np.array([
    0.15447199836826409, -0.04178051880210157, 0.254740941008832, -0.030609540254954375,
    0.0364027796309115, 0.02461986794302449, 0.009428584990410598,
    0.1707905908904179, 0.014105150014657371, 0.2443540287559067, -0.0680544885409485,
    0.5084641251535916, -0.003864834389433353, 0.2332398898472048
], dtype=np.float32)

EE_POSE_STD = np.array([
    0.1985289482817342, 0.0829915429598024, 0.06982858867939001, 0.08557872418466743,
    0.8731148881898303, 0.08805723262926324, 0.46865267306593383,
    0.2036059192693966, 0.0985647436467834, 0.0698404568789507, 0.12701774133881905,
    0.7124046333088213, 0.03690152950585441, 0.396738574984529
], dtype=np.float32)

# Base Velocity 归一化参数 (observation.base_velocity) - 3D
BASE_VELOCITY_MEAN = np.array([
    -3.3837676765214757e-07, 0.0, -3.0204660877970014e-07
], dtype=np.float32)

BASE_VELOCITY_STD = np.array([
    2.949571593712036e-05, 1e-6, 2.8592489908776558e-05  # 避免除零
], dtype=np.float32)

# Action 反归一化参数 - 17D (前3D是base velocity, 后14D是arm joints)
# 注意: 前3D的std=0，模型使用 action_dim_offset=3 跳过
ACTION_MEAN = np.array([
    0.0, -0.00057220458984375, 0.0,  # base velocity (跳过)
    -0.258587418424181, 0.7454151136628402, -0.9228769868890331, 0.02938408466256687,
    0.7823781220945318, -0.1229191926611769, 0.06786235497213942,
    0.0920435152866867, 0.8162668512498173, -0.9697007612362752, -0.04376927294546041,
    0.8052032868458314, -0.06241027563742915, 0.06792804105653465
], dtype=np.float32)

ACTION_STD = np.array([
    1e-6, 1e-6, 1e-6,  # base velocity (跳过, 避免除零)
    0.14637404692389755, 0.9804744205436965, 0.6288037455782007, 0.1720257952922301,
    0.15680759971719502, 0.15103698248425998, 0.028401691950683535,
    0.21647223403427238, 1.032588633388957, 0.7129228058021149, 0.08252845396558328,
    0.1411880111771137, 0.17439126979523825, 0.022577296610396563
], dtype=np.float32)

# 只取后14D的action参数 (跳过前3D base velocity)
ACTION_MEAN_14D = ACTION_MEAN[3:]
ACTION_STD_14D = ACTION_STD[3:]

# Image 归一化参数（ImageNet 标准）
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------- Global buffers -----------------
latest_imgs = {
    "main": None,
    "secondary_0": None,  # wrist_left
    "secondary_1": None,  # wrist_right
    "secondary_2": None,  # 可复用main或其他相机
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

latest_ee_pose = {
    "left": None,
    "right": None,
}

latest_base_velocity = None

smoothed_action = {
    "left": None,
    "right": None,
}


# ----------------- Helper functions -----------------
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


# ----------------- Image callbacks -----------------
def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)

def cb_secondary_0(msg: CompressedImage):
    latest_imgs["secondary_0"] = decode_compressed_image(msg)

def cb_secondary_1(msg: CompressedImage):
    latest_imgs["secondary_1"] = decode_compressed_image(msg)

def cb_secondary_2(msg: CompressedImage):
    latest_imgs["secondary_2"] = decode_compressed_image(msg)


# ----------------- Joint callbacks -----------------
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


# ----------------- End effector pose callbacks -----------------
def cb_ee_pose_left(msg: PoseStamped):
    latest_ee_pose["left"] = np.array([
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z, msg.pose.orientation.w
    ], dtype=np.float32)

def cb_ee_pose_right(msg: PoseStamped):
    latest_ee_pose["right"] = np.array([
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z, msg.pose.orientation.w
    ], dtype=np.float32)


# ----------------- Base velocity callback -----------------
def cb_base_velocity(msg: Twist):
    global latest_base_velocity
    latest_base_velocity = np.array([
        msg.linear.x, msg.linear.y, msg.angular.z
    ], dtype=np.float32)


# ----------------- Preprocessing functions -----------------
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


def normalize_effort(effort: np.ndarray) -> np.ndarray:
    return (effort - EFFORT_MEAN) / EFFORT_STD


def normalize_velocity(velocity: np.ndarray) -> np.ndarray:
    return (velocity - VELOCITY_MEAN) / VELOCITY_STD


def normalize_ee_pose(ee_pose: np.ndarray) -> np.ndarray:
    return (ee_pose - EE_POSE_MEAN) / EE_POSE_STD


def normalize_base_velocity(base_vel: np.ndarray) -> np.ndarray:
    return (base_vel - BASE_VELOCITY_MEAN) / BASE_VELOCITY_STD


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Unnormalize 14D action output."""
    return action * ACTION_STD_14D + ACTION_MEAN_14D


# ----------------- Load ACT policy -----------------
def load_policy(ckpt_dir: str, device: str) -> ACTPolicy:
    # Check if it's a local path that exists
    local_path = Path(ckpt_dir).expanduser().resolve()
    if local_path.exists():
        pretrained_path = str(local_path)
        rospy.loginfo(f"Loading ACT Policy from local path: {pretrained_path}")
    else:
        # Assume it's a Hugging Face repo ID
        pretrained_path = ckpt_dir
        rospy.loginfo(f"Loading ACT Policy from Hugging Face: {pretrained_path}")

    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=pretrained_path)
    policy = policy.to(device)

    rospy.loginfo("=" * 70)
    rospy.loginfo("[INFO] Policy loaded successfully")
    rospy.loginfo(f"  use_torque: {policy.config.use_torque}")
    rospy.loginfo(f"  action_dim_offset: {policy.config.action_dim_offset}")
    rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
    rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
    rospy.loginfo("=" * 70)

    policy.eval()
    return policy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ACT Policy Deployment with Torque Support")
    parser.add_argument("--model", type=str, default="torque-true",
                        choices=["torque-false", "torque-true", "local"],
                        help="Model to use: torque-false, torque-true, or local")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Local checkpoint path (required if --model=local)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA smoothing alpha (0=no smoothing, 1=no history)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable EMA smoothing")

    # ROS初始化前解析参数
    args, unknown = parser.parse_known_args()

    rospy.init_node("piper_act_wholebody_torque")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")

    # 选择模型
    if args.model == "torque-false":
        ckpt_dir = "/home/zeno/ACT-wholebody/ACT-wholebody/checkpoints/ACT-100-wholebody-torque-false"
        rospy.loginfo("Using model: torque=false (baseline, no torque info)")
    elif args.model == "torque-true":
        ckpt_dir = "/home/zeno/ACT-wholebody/ACT-wholebody/checkpoints/ACT-100-wholebody-torque-true"
        rospy.loginfo("Using model: torque=true (with torque info)")
    else:
        if args.ckpt is None:
            rospy.logerr("--ckpt is required when --model=local")
            return
        ckpt_dir = args.ckpt
        rospy.loginfo(f"Using local model: {ckpt_dir}")

    policy = load_policy(ckpt_dir, device)
    policy.reset()

    use_torque = policy.config.use_torque

    # ----------------- ROS subs / pubs -----------------
    # 图像订阅
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_secondary_0, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_secondary_1, queue_size=1)
    # secondary_2 可以订阅另一个相机，或者在代码中复用main
    # rospy.Subscriber("/realsense_other/color/image_raw/compressed", CompressedImage, cb_secondary_2, queue_size=1)

    # 关节状态订阅 (包含position, velocity, effort)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # 末端执行器位姿订阅
    rospy.Subscriber("/robot/arm_left/end_pose", PoseStamped, cb_ee_pose_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/end_pose", PoseStamped, cb_ee_pose_right, queue_size=1)

    # 底盘速度订阅
    rospy.Subscriber("/robot/base/velocity", Twist, cb_base_velocity, queue_size=1)

    # 发布器
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    rate = rospy.Rate(args.rate)

    ENABLE_SMOOTHING = not args.no_smoothing
    SMOOTHING_ALPHA = args.smoothing

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] Deployment settings:")
    rospy.loginfo(f"  Model: {args.model}")
    rospy.loginfo(f"  use_torque: {use_torque}")
    rospy.loginfo(f"  Control rate: {args.rate} Hz")
    rospy.loginfo(f"  EMA smoothing: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0

    while not rospy.is_shutdown():
        # 检查必要数据是否就绪
        # 图像: main, secondary_0, secondary_1 必须有
        if (latest_imgs["main"] is None or
            latest_imgs["secondary_0"] is None or
            latest_imgs["secondary_1"] is None):
            rate.sleep()
            continue

        # 关节状态必须有
        if latest_q["left"] is None or latest_q["right"] is None:
            rate.sleep()
            continue

        # 如果使用torque模型，effort必须有
        if use_torque and (latest_effort["left"] is None or latest_effort["right"] is None):
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("✓ All required sensors ready")
            data_ready_logged = True

        # --------------- Build observation dict ---------------
        # 图像预处理
        main_img = preprocess_image(latest_imgs["main"]).to(device)
        secondary_0 = preprocess_image(latest_imgs["secondary_0"]).to(device)
        secondary_1 = preprocess_image(latest_imgs["secondary_1"]).to(device)
        # secondary_2: 如果没有第4个相机，复用main
        if latest_imgs["secondary_2"] is not None:
            secondary_2 = preprocess_image(latest_imgs["secondary_2"]).to(device)
        else:
            secondary_2 = main_img

        # State 归一化 (14D)
        state_raw = np.concatenate([latest_q["left"], latest_q["right"]], axis=0).astype(np.float32)
        state_normalized = normalize_state(state_raw)
        state = torch.from_numpy(state_normalized[None, :]).to(device)

        # Effort/Torque 归一化 (14D)
        if latest_effort["left"] is not None and latest_effort["right"] is not None:
            effort_raw = np.concatenate([latest_effort["left"], latest_effort["right"]], axis=0).astype(np.float32)
        else:
            effort_raw = np.zeros(14, dtype=np.float32)
        effort_normalized = normalize_effort(effort_raw)
        effort = torch.from_numpy(effort_normalized[None, :]).to(device)

        # Velocity 归一化 (14D)
        if latest_velocity["left"] is not None and latest_velocity["right"] is not None:
            velocity_raw = np.concatenate([latest_velocity["left"], latest_velocity["right"]], axis=0).astype(np.float32)
        else:
            velocity_raw = np.zeros(14, dtype=np.float32)
        velocity_normalized = normalize_velocity(velocity_raw)
        velocity = torch.from_numpy(velocity_normalized[None, :]).to(device)

        # EE Pose 归一化 (14D)
        if latest_ee_pose["left"] is not None and latest_ee_pose["right"] is not None:
            ee_pose_raw = np.concatenate([latest_ee_pose["left"], latest_ee_pose["right"]], axis=0).astype(np.float32)
        else:
            ee_pose_raw = np.zeros(14, dtype=np.float32)
        ee_pose_normalized = normalize_ee_pose(ee_pose_raw)
        ee_pose = torch.from_numpy(ee_pose_normalized[None, :]).to(device)

        # Base Velocity 归一化 (3D)
        if latest_base_velocity is not None:
            base_vel_raw = latest_base_velocity
        else:
            base_vel_raw = np.zeros(3, dtype=np.float32)
        base_vel_normalized = normalize_base_velocity(base_vel_raw)
        base_velocity = torch.from_numpy(base_vel_normalized[None, :]).to(device)

        # 构建观测字典
        obs = {
            "observation.images.main": main_img,
            "observation.images.secondary_0": secondary_0,
            "observation.images.secondary_1": secondary_1,
            "observation.images.secondary_2": secondary_2,
            "observation.state": state,
            "observation.effort": effort,
            "observation.velocity": velocity,
            "observation.ee_pose": ee_pose,
            "observation.base_velocity": base_velocity,
        }

        # --------------- Policy inference ---------------
        with torch.no_grad():
            action_tensor = policy.select_action(obs)

        if action_tensor.dim() == 2:
            action_normalized = action_tensor[0, :].cpu().numpy()
        else:
            action_normalized = action_tensor.cpu().numpy()

        # Action 反归一化 (14D)
        action = unnormalize_action(action_normalized)

        if len(action) != 14:
            rospy.logwarn(f"Invalid action dim: {len(action)}, expected 14")
            rate.sleep()
            continue

        action_left = action[:7].copy()
        action_right = action[7:14].copy()

        # --------------- Diagnostic logging ---------------
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count}")
            rospy.loginfo(f"  Raw state LEFT:        {np.array2string(latest_q['left'], precision=3)}")
            rospy.loginfo(f"  Raw state RIGHT:       {np.array2string(latest_q['right'], precision=3)}")
            if use_torque:
                rospy.loginfo(f"  Raw effort LEFT:       {np.array2string(latest_effort['left'], precision=3)}")
                rospy.loginfo(f"  Raw effort RIGHT:      {np.array2string(latest_effort['right'], precision=3)}")
            rospy.loginfo(f"  Model output (norm):   {np.array2string(action_normalized[:7], precision=3)}...")
            rospy.loginfo(f"  Unnormalized action L: {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Unnormalized action R: {np.array2string(action_right, precision=3)}")

            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]
            rospy.loginfo(f"  Delta LEFT:  {np.array2string(delta_left, precision=3)}")
            rospy.loginfo(f"  Delta RIGHT: {np.array2string(delta_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # --------------- EMA smoothing ---------------
        global smoothed_action
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
            else:
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]

        # --------------- Publish ---------------
        # 关节名称 (Piper 7-DOF arm)
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()

        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()

        pub_left.publish(msg_left)
        pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, f"✓ Actions sent (torque={use_torque})")

        rate.sleep()


if __name__ == "__main__":
    main()
