from pathlib import Path
import shutil
import numpy as np
import cv2
import time
import os
from multiprocessing import Manager

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ============================================================
# DATA SOURCE CONFIGURATION
# ============================================================
DATA_SOURCE = "local"
DATA_ROOT = Path("/workspace/ACT-wholebody-torque/ACT-fullbody/data/ACT-120")
HF_LEROBOT_HOME = Path("/workspace/ACT-wholebody-torque/ACT-fullbody/data")
REPO_NAME = "ACT-120-V30"
TASK_LABEL = "Place the basket on the red marker and pick the yellow pepper into the basket."

# ============================================================
# ROS TOPICS
# ============================================================
# Cameras
CAM_WRIST_LEFT = "/realsense_left/color/image_raw/compressed"
CAM_WRIST_RIGHT = "/realsense_right/color/image_raw/compressed"
CAM_WIDE_TOP = "/realsense_top/color/image_raw/compressed"
CAM_MAIN = CAM_WIDE_TOP

# Arm states and actions
STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"

# Mobile base (NEW)
ODOM_TOPIC = "/ranger_base_node/odom"
TELEOP_CMD_VEL = "/teleop/cmd_vel"

# Optional end poses
END_POSE_LEFT = "/robot/arm_left/end_pose"
END_POSE_RIGHT = "/robot/arm_right/end_pose"

# ============================================================
# SETTINGS
# ============================================================
FPS = 10
IMG_SIZE = (224, 224)
NUM_WORKERS = 1


def decode_compressed_image(msg):
    """sensor_msgs/CompressedImage -> HxWx3 uint8 RGB (resized)."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def nearest_idx(times, t):
    """Find nearest timestamp index."""
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = times[idx - 1]
    after = times[idx]
    return idx if abs(after - t) < abs(t - before) else idx - 1


def quaternion_to_yaw(quat):
    """Convert quaternion to yaw angle (orientation.z, orientation.w)."""
    # quat: geometry_msgs/Quaternion with x, y, z, w
    # Yaw (z-axis rotation) = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return yaw


def collect_bag_files():
    """Collect all .bag files from local directory."""
    if not DATA_ROOT.exists():
        print(f"Warning: Data directory not found: {DATA_ROOT}")
        return []

    bag_files = sorted(DATA_ROOT.glob("*.bag"))
    if not bag_files:
        print(f"Warning: No .bag files found in {DATA_ROOT}")
    else:
        print(f"Found {len(bag_files)} bag file(s) in {DATA_ROOT}:")
        for bag_file in bag_files:
            print(f"  - {bag_file.name}")

    return bag_files


def process_single_bag(args):
    """Process a single bag file and return episode data."""
    bag_path, task_name, bag_idx, total_bags, progress_dict, lock = args

    with lock:
        print(f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path.name}")

    bag_start_time = time.time()

    try:
        with AnyReader([bag_path]) as reader:
            interested_topics = {
                CAM_MAIN, CAM_WRIST_LEFT, CAM_WRIST_RIGHT, CAM_WIDE_TOP,
                STATE_LEFT, STATE_RIGHT, ACTION_LEFT, ACTION_RIGHT,
                ODOM_TOPIC, TELEOP_CMD_VEL,  # NEW
                END_POSE_LEFT, END_POSE_RIGHT,
            }
            topic_to_msgs = {topic: [] for topic in interested_topics}

            connections = [c for c in reader.connections if c.topic in interested_topics]
            if not connections:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: No relevant topics found, skipping.")
                return None

            for conn, t, raw in reader.messages(connections=connections):
                if conn.topic not in topic_to_msgs:
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                topic_to_msgs[conn.topic].append((t, msg))

            # Extract messages
            cam_main_msgs = topic_to_msgs[CAM_MAIN]
            cam_wrist_left_msgs = topic_to_msgs[CAM_WRIST_LEFT]
            cam_wrist_right_msgs = topic_to_msgs[CAM_WRIST_RIGHT]
            cam_wide_top_msgs = topic_to_msgs[CAM_WIDE_TOP]
            state_left_msgs = topic_to_msgs[STATE_LEFT]
            state_right_msgs = topic_to_msgs[STATE_RIGHT]
            action_left_msgs = topic_to_msgs[ACTION_LEFT]
            action_right_msgs = topic_to_msgs[ACTION_RIGHT]
            odom_msgs = topic_to_msgs[ODOM_TOPIC]  # NEW
            cmd_vel_msgs = topic_to_msgs[TELEOP_CMD_VEL]  # NEW

            # Validation
            if not cam_main_msgs or not cam_wrist_left_msgs:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing camera data, skipping")
                return None
            if not state_left_msgs or not state_right_msgs:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing state data, skipping")
                return None
            if not action_left_msgs or not action_right_msgs:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing action data, skipping")
                return None
            if not odom_msgs:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing odom data, skipping")
                return None
            if not cmd_vel_msgs:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing cmd_vel data, skipping")
                return None

            # Extract timestamps
            cam_main_times = np.array([t for t, _ in cam_main_msgs], dtype=np.int64)
            wrist_left_times = np.array([t for t, _ in cam_wrist_left_msgs], dtype=np.int64)
            wrist_right_times = np.array([t for t, _ in cam_wrist_right_msgs], dtype=np.int64)
            wide_top_times = np.array([t for t, _ in cam_wide_top_msgs], dtype=np.int64)
            state_left_times = np.array([t for t, _ in state_left_msgs], dtype=np.int64)
            state_right_times = np.array([t for t, _ in state_right_msgs], dtype=np.int64)
            action_left_times = np.array([t for t, _ in action_left_msgs], dtype=np.int64)
            action_right_times = np.array([t for t, _ in action_right_msgs], dtype=np.int64)
            odom_times = np.array([t for t, _ in odom_msgs], dtype=np.int64)  # NEW
            cmd_vel_times = np.array([t for t, _ in cmd_vel_msgs], dtype=np.int64)  # NEW

            # Find common time range
            t_start = max(cam_main_times[0], wrist_left_times[0], state_left_times[0])
            t_end = min(cam_main_times[-1], wrist_left_times[-1], state_left_times[-1])

            if t_end <= t_start:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Non-positive duration, skipping")
                return None

            common_duration = (t_end - t_start) / 1e9
            with lock:
                print(f"[Bag {bag_idx}/{total_bags}] Common time range: {common_duration:.2f} seconds")

            # Generate uniform timestamps
            min_dt = int(1e9 / FPS)
            num_frames = int((t_end - t_start) / min_dt)
            if num_frames <= 0:
                with lock:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: num_frames <= 0, skipping")
                return None

            uniform_timestamps = np.linspace(t_start, t_end, num_frames, dtype=np.int64)

            with lock:
                print(f"[Bag {bag_idx}/{total_bags}] Generating {num_frames} frames at {FPS} FPS")

            # Process frames
            episode_frames = []
            start_time = time.time()

            for frame_count, t_frame in enumerate(uniform_timestamps, 1):
                # Images
                idx_main = nearest_idx(cam_main_times, t_frame)
                idx_wl = nearest_idx(wrist_left_times, t_frame)
                idx_wr = nearest_idx(wrist_right_times, t_frame)
                idx_wt = nearest_idx(wide_top_times, t_frame)

                main_image = decode_compressed_image(cam_main_msgs[idx_main][1])
                wrist_left_image = decode_compressed_image(cam_wrist_left_msgs[idx_wl][1])
                wrist_right_image = decode_compressed_image(cam_wrist_right_msgs[idx_wr][1])
                wide_top_image = decode_compressed_image(cam_wide_top_msgs[idx_wt][1])

                # States
                idx_sl = nearest_idx(state_left_times, t_frame)
                idx_sr = nearest_idx(state_right_times, t_frame)
                state_left_msg = state_left_msgs[idx_sl][1]
                state_right_msg = state_right_msgs[idx_sr][1]

                # Extract position, velocity, effort (14D each)
                state_left_pos = np.array(state_left_msg.position, dtype=np.float32)[:7]
                state_right_pos = np.array(state_right_msg.position, dtype=np.float32)[:7]
                state_left_vel = np.array(state_left_msg.velocity, dtype=np.float32)[:7]
                state_right_vel = np.array(state_right_msg.velocity, dtype=np.float32)[:7]
                state_left_eff = np.array(state_left_msg.effort, dtype=np.float32)[:7]
                state_right_eff = np.array(state_right_msg.effort, dtype=np.float32)[:7]

                state_14d = np.concatenate([state_left_pos, state_right_pos])  # 14D
                velocity_14d = np.concatenate([state_left_vel, state_right_vel])  # 14D
                effort_14d = np.concatenate([state_left_eff, state_right_eff])  # 14D

                # Odom (base velocity: vx, vy, omega)
                idx_odom = nearest_idx(odom_times, t_frame)
                odom_msg = odom_msgs[idx_odom][1]
                base_vx = float(odom_msg.twist.twist.linear.x)
                base_vy = float(odom_msg.twist.twist.linear.y)
                base_omega = float(odom_msg.twist.twist.angular.z)
                base_velocity = np.array([base_vx, base_vy, base_omega], dtype=np.float32)  # 3D

                # Actions
                idx_al = nearest_idx(action_left_times, t_frame)
                idx_ar = nearest_idx(action_right_times, t_frame)
                action_left_msg = action_left_msgs[idx_al][1]
                action_right_msg = action_right_msgs[idx_ar][1]

                action_left = np.array(action_left_msg.position, dtype=np.float32)[:7]
                action_right = np.array(action_right_msg.position, dtype=np.float32)[:7]

                # Cmd_vel (base action: vx, vy, omega)
                idx_cmd = nearest_idx(cmd_vel_times, t_frame)
                cmd_vel_msg = cmd_vel_msgs[idx_cmd][1]
                action_base_vx = float(cmd_vel_msg.linear.x)
                action_base_vy = float(cmd_vel_msg.linear.y)
                action_base_omega = float(cmd_vel_msg.angular.z)
                action_base = np.array([action_base_vx, action_base_vy, action_base_omega], dtype=np.float32)  # 3D

                # Assemble 17D action: [base_vx, base_vy, base_omega, left_arm, right_arm]
                action_17d = np.concatenate([action_base, action_left, action_right])  # 17D

                # Build frame
                frame = {
                    "observation.images.main": main_image,
                    "observation.images.secondary_0": wrist_left_image,
                    "observation.images.secondary_1": wrist_right_image,
                    "observation.images.secondary_2": wide_top_image,
                    "observation.state": state_14d,  # 14D (dual-arm only)
                    "observation.velocity": velocity_14d,  # 14D (dual-arm velocity)
                    "observation.effort": effort_14d,  # 14D (dual-arm torque)
                    "observation.base_velocity": base_velocity,  # 3D (base velocity)
                    "action": action_17d,  # 17D (base + dual-arm)
                    "task": task_name,
                }

                episode_frames.append(frame)

            elapsed = time.time() - start_time
            with lock:
                print(f"  [Bag {bag_idx}/{total_bags}] ✓ Completed in {elapsed:.1f}s ({num_frames} frames)")
                progress_dict["completed"] += 1
                print(f"  Progress: {progress_dict['completed']}/{total_bags} bags completed")

            return episode_frames

    except Exception as e:
        with lock:
            print(f"[Bag {bag_idx}/{total_bags}] Error: {e}")
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    total_start_time = time.time()

    # Create dataset
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        "action": {
            "dtype": "float32",
            "shape": (17,),
            "names": ["base_vx", "base_vy", "base_omega"]
                     + [f"left_joint_{i}" for i in range(7)]
                     + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}" for i in range(7)]
                     + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}_vel" for i in range(7)]
                     + [f"right_joint_{i}_vel" for i in range(7)],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}_eff" for i in range(7)]
                     + [f"right_joint_{i}_eff" for i in range(7)],
        },
        "observation.base_velocity": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["base_vx", "base_vy", "base_omega"],
        },
        "observation.images.main": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_0": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_1": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_2": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="zeno",
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=4,
    )

    print(f"\n{'=' * 60}")
    print(f"Converting to ACT-wholebody format (17D actions)")
    print(f"Task: {TASK_LABEL}")
    print(f"Data path: {DATA_ROOT}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    bag_files = collect_bag_files()
    total_bags = len(bag_files)

    if total_bags == 0:
        print("No bag files to process.")
    else:
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict["completed"] = 0
        lock = manager.Lock()

        bag_args = [
            (bag_path, TASK_LABEL, bag_idx, total_bags, progress_dict, lock)
            for bag_idx, bag_path in enumerate(bag_files, 1)
        ]

        print(f"\nProcessing {total_bags} bags...")

        results = []
        for args in bag_args:
            results.append(process_single_bag(args))

        print(f"\n{'=' * 60}")
        print("Adding episodes to dataset...")
        print(f"{'=' * 60}")

        successful_episodes = 0
        for result in results:
            if result is not None:
                for frame in result:
                    dataset.add_frame(frame)
                dataset.save_episode()
                successful_episodes += 1

        # dataset.consolidate()  # Not needed in newer LeRobot versions

        total_elapsed = time.time() - total_start_time
        print(f"\n{'=' * 60}")
        print(f"✓ Conversion complete!")
        print(f"  Successfully converted: {successful_episodes}/{total_bags} episodes")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}\n")
