import yaml
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from lerobot_robot_ur5e import UR5eConfig, UR5e, DualUR5e
from lerobot_teleoperator_ur5e import UR5eTeleopConfig, UR5eTeleop
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
import shutil
import termios, sys
from lerobot.utils.constants import HF_LEROBOT_HOME
from scripts.utils.teleop_joint_offsets import get_start_joints, compute_joint_offsets, save_calibration
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

class RecordConfig:
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        teleop = cfg["teleop"]

        # global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)

        # Check if dual-arm mode is enabled from config
        self.is_dual_arm = cfg.get("dual_arm_mode", False)

        if self.is_dual_arm:
            # Dual-arm teleop config
            self.port = teleop["port"]
            self.control_mode = teleop.get("control_mode", "isoteleop")
            self.left_arm = teleop["left_arm"]
            self.right_arm = teleop["right_arm"]

            # Dual-arm robot config
            self.robot_left = robot["left_arm"]
            self.robot_right = robot["right_arm"]

            # Dual-arm cameras config
            self.left_wrist_cam_serial: str = cam["left_wrist_cam_serial"]
            self.left_exterior_cam_serial: str = cam["left_exterior_cam_serial"]
            self.right_wrist_cam_serial: str = cam["right_wrist_cam_serial"]
            self.right_exterior_cam_serial: str = cam["right_exterior_cam_serial"]
        else:
            # Single-arm config (backward compatibility)
            dxl_cfg = teleop["dynamixel_config"]
            self.port = dxl_cfg["port"]
            self.use_gripper = dxl_cfg["use_gripper"]
            self.joint_ids = dxl_cfg["joint_ids"]
            self.joint_offsets = dxl_cfg["joint_offsets"]
            self.joint_signs = dxl_cfg["joint_signs"]
            self.gripper_config = dxl_cfg["gripper_config"]
            self.hardware_offsets = dxl_cfg["hardware_offsets"]
            self.control_mode = teleop.get("control_mode", "isoteleop")

            # Single-arm robot config
            self.robot_ip: str = robot["ip"]
            self.start_position = robot.get("start_position", [0, -30, 60, -100, 130, 0])
            self.gripper_port: str = robot["gripper_port"]
            self.use_gripper: str = robot["use_gripper"]
            self.close_threshold = robot["close_threshold"]
            self.gripper_reverse: str = robot["gripper_reverse"]
            self.gripper_bin_threshold: float = robot["gripper_bin_threshold"]

            # Single-arm cameras config
            self.wrist_cam_serial: str = cam["wrist_cam_serial"]
            self.exterior_cam_serial: str = cam["exterior_cam_serial"]

        self.width: int = cam["width"]
        self.height: int = cam["height"]

        # task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", "False")
        self.resume_dataset: str = task["resume_dataset"]

        # time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        self.save_mera_period: int = time.get("save_mera_period", 1)

        # storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)


def check_joint_offsets(record_cfg: RecordConfig, arm_name: str = None):
    """Check the joint_offsets is set and correct."""

    if record_cfg.is_dual_arm:
        # For dual-arm, we need to check the specific arm
        if arm_name == "left":
            joint_offsets_to_check = record_cfg.left_arm["joint_offsets"]
            hardware_offsets_to_check = record_cfg.left_arm["hardware_offsets"]
            robot_ip = record_cfg.robot_left["ip"]
        else:
            joint_offsets_to_check = record_cfg.right_arm["joint_offsets"]
            hardware_offsets_to_check = record_cfg.right_arm["hardware_offsets"]
            robot_ip = record_cfg.robot_right["ip"]
    else:
        joint_offsets_to_check = record_cfg.joint_offsets
        hardware_offsets_to_check = record_cfg.hardware_offsets
        robot_ip = record_cfg.robot_ip

    if joint_offsets_to_check is None:
        raise ValueError("joint_offsets is None. Please check teleop_joint_offsets.py output.")

    # Create a temporary config object for get_start_joints
    if record_cfg.is_dual_arm:
        if arm_name == "left":
            start_pos = record_cfg.robot_left.get("start_position", [0, -30, 60, -100, 130, 0])
        else:
            start_pos = record_cfg.robot_right.get("start_position", [0, -30, 60, -100, 130, 0])
        temp_cfg = type('obj', (object,), {'robot_ip': robot_ip, 'start_position': start_pos})()
    else:
        temp_cfg = type('obj', (object,), {'robot_ip': robot_ip, 'start_position': record_cfg.start_position})()
    start_joints = get_start_joints(temp_cfg)
    if start_joints is None:
        raise RuntimeError("Failed to retrieve start joints from UR5e robot.")

    # Create another temp config for compute_joint_offsets
    if record_cfg.is_dual_arm:
        if arm_name == "left":
            temp_cfg = type('obj', (object,), {
                'joint_ids': record_cfg.left_arm["joint_ids"],
                'joint_signs': record_cfg.left_arm["joint_signs"],
                'hardware_offsets': record_cfg.left_arm["hardware_offsets"],
                'port': record_cfg.port
            })()
        else:
            temp_cfg = type('obj', (object,), {
                'joint_ids': record_cfg.right_arm["joint_ids"],
                'joint_signs': record_cfg.right_arm["joint_signs"],
                'hardware_offsets': record_cfg.right_arm["hardware_offsets"],
                'port': record_cfg.port
            })()
    else:
        temp_cfg = record_cfg

    joint_offsets = compute_joint_offsets(temp_cfg, start_joints)

    if joint_offsets != joint_offsets_to_check:
        logging.error(f"====== [ERROR] Computed joint_offsets {joint_offsets} != provided joint_offsets {joint_offsets_to_check}. Please check teleop_joint_offsets.py output. ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to update with the new computed values and retry? (y/n): ").strip().lower()

        if ans == "y":
            logging.info(f"====== [UPDATE] Updating joint_offsets from {joint_offsets_to_check} to {joint_offsets} ======")
            parent_path = Path(__file__).resolve().parent
            cfg_path = parent_path.parent / "config" / "cfg.yaml"

            # For dual-arm, pass the arm name to save_calibration
            arm_name_for_save = f"{arm_name}_arm" if record_cfg.is_dual_arm else None
            save_calibration(cfg_path, hardware_offsets_to_check, joint_offsets, arm_name_for_save)

            # Update the config
            if record_cfg.is_dual_arm:
                if arm_name == "left":
                    record_cfg.left_arm["joint_offsets"] = joint_offsets
                else:
                    record_cfg.right_arm["joint_offsets"] = joint_offsets
            else:
                record_cfg.joint_offsets = joint_offsets

            # Re-check with the updated values
            return check_joint_offsets(record_cfg, arm_name)
        else:
            raise ValueError(
                f"Joint offset mismatch not resolved. Computed: {joint_offsets}, Provided: {joint_offsets_to_check}. "
                "Please run teleop_joint_offsets.py to get the correct values."
            )
    logging.info("Joint offsets verified successfully.")

def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        print(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            print(f"====== [DELETE] Removing folder: {dataset_path} ======")
            shutil.rmtree(dataset_path, ignore_errors=True)  # Delete only this specific dataset folder
            print("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            print("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")

def enter_freedrive(robot, is_dual_arm):
    if is_dual_arm:
        try:
            robot.left_arm._arm["rtde_c"].servoStop()
        except Exception:
            pass
        try:
            robot.right_arm._arm["rtde_c"].servoStop()
        except Exception:
            pass
        time.sleep(0.2)
        robot.left_arm._arm["rtde_c"].stopScript()
        robot.right_arm._arm["rtde_c"].stopScript()
        time.sleep(0.3)
        robot.left_arm._arm["rtde_c"].reuploadScript()
        robot.right_arm._arm["rtde_c"].reuploadScript()
        time.sleep(0.3)
        ok_l = robot.left_arm._arm["rtde_c"].freedriveMode()
        ok_r = robot.right_arm._arm["rtde_c"].freedriveMode()
        logging.info(f"[FREEDRIVE] freedriveMode returned: left={ok_l}, right={ok_r}")
    else:
        try:
            robot._arm["rtde_c"].servoStop()
        except Exception:
            pass
        time.sleep(0.2)
        robot._arm["rtde_c"].stopScript()
        time.sleep(0.3)
        robot._arm["rtde_c"].reuploadScript()
        time.sleep(0.3)
        ok = robot._arm["rtde_c"].freedriveMode()
        logging.info(f"[FREEDRIVE] freedriveMode returned: {ok}")

def exit_freedrive_and_return(robot, is_dual_arm):
    if is_dual_arm:
        robot.left_arm._arm["rtde_c"].endFreedriveMode()
        robot.right_arm._arm["rtde_c"].endFreedriveMode()
        time.sleep(0.2)
        robot.left_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.left_arm.config.start_position).tolist(), 0.5, 0.5)
        robot.right_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.right_arm.config.start_position).tolist(), 0.5, 0.5)
    else:
        robot._arm["rtde_c"].endFreedriveMode()
        time.sleep(0.2)
        robot._arm["rtde_c"].moveJ(np.deg2rad(robot.config.start_position).tolist(), 0.5, 0.5)

def exit_freedrive_only(robot, is_dual_arm):
    """Exit freedrive without returning to start position (for transitioning to teleop sub-mode)."""
    if is_dual_arm:
        robot.left_arm._arm["rtde_c"].endFreedriveMode()
        robot.right_arm._arm["rtde_c"].endFreedriveMode()
    else:
        robot._arm["rtde_c"].endFreedriveMode()
    time.sleep(0.2)

def _read_ur5e_joints(robot, is_dual_arm):
    """Read current UR5e joint positions."""
    if is_dual_arm:
        left_q = robot.left_arm._arm["rtde_r"].getActualQ()
        right_q = robot.right_arm._arm["rtde_r"].getActualQ()
        return left_q, right_q
    else:
        return robot._arm["rtde_r"].getActualQ()

def _servo_ur5e(robot, joint_positions, is_dual_arm):
    """Send servoJ command to UR5e."""
    dt = 0.002
    lookahead = 0.2
    gain = 100
    vel = 0.5
    accel = 0.5
    if is_dual_arm:
        left_pos, right_pos = joint_positions
        t_l = robot.left_arm._arm["rtde_c"].initPeriod()
        robot.left_arm._arm["rtde_c"].servoJ(list(left_pos), vel, accel, dt, lookahead, gain)
        robot.left_arm._arm["rtde_c"].waitPeriod(t_l)
        t_r = robot.right_arm._arm["rtde_c"].initPeriod()
        robot.right_arm._arm["rtde_c"].servoJ(list(right_pos), vel, accel, dt, lookahead, gain)
        robot.right_arm._arm["rtde_c"].waitPeriod(t_r)
    else:
        t_start = robot._arm["rtde_c"].initPeriod()
        robot._arm["rtde_c"].servoJ(list(joint_positions), vel, accel, dt, lookahead, gain)
        robot._arm["rtde_c"].waitPeriod(t_start)

def handle_freedrive_if_requested(freedrive_state, events, robot, is_dual_arm, teleop, dataset):
    if not freedrive_state["request_enter"]:
        return False
    freedrive_state["request_enter"] = False
    events["exit_early"] = False

    logging.info("====== [FREEDRIVE] Entering freedrive mode with mArm following. ======")
    logging.info("  'i' = teleop sub-mode | 'o' = back to freedrive | 'b' = exit freedrive")

    # Enter freedrive: UR5e freedrive + mArm stiff tracking
    enter_freedrive(robot, is_dual_arm)
    teleop.set_marm_pid_profile("stiff")
    teleop.set_marm_torque(True)
    freedrive_state["active"] = True
    freedrive_state["sub_mode"] = "freedrive"

    try:
        while not freedrive_state["request_exit"] and not events["stop_recording"]:
            if freedrive_state["sub_mode"] == "freedrive":
                # mArm follows UR5e
                try:
                    if is_dual_arm:
                        left_q, right_q = _read_ur5e_joints(robot, True)
                        teleop.command_marm_from_ur5e(left_q[:6], arm="left")
                        teleop.command_marm_from_ur5e(right_q[:6], arm="right")
                    else:
                        ur5e_q = _read_ur5e_joints(robot, False)
                        teleop.command_marm_from_ur5e(ur5e_q[:6])
                except RuntimeError as e:
                    logging.warning(f"[FREEDRIVE] mArm command failed (transient): {e}")

                # Check for teleop sub-mode request
                if freedrive_state.get("request_teleop"):
                    freedrive_state["request_teleop"] = False
                    logging.info("[FREEDRIVE] Switching to teleop sub-mode (mArm soft, UR5e follows mArm)...")
                    exit_freedrive_only(robot, is_dual_arm)
                    teleop.set_marm_pid_profile("soft")
                    freedrive_state["sub_mode"] = "teleop"
                    logging.info("[FREEDRIVE] Teleop sub-mode active. 'o' to return to freedrive, 'b' to exit.")

                time.sleep(0.02)  # ~50Hz

            elif freedrive_state["sub_mode"] == "teleop":
                # UR5e follows mArm (normal teleop direction)
                try:
                    if is_dual_arm:
                        left_state = teleop.get_marm_joint_state(arm="left")
                        right_state = teleop.get_marm_joint_state(arm="right")
                        _servo_ur5e(robot, (left_state[:6].tolist(), right_state[:6].tolist()), True)
                    else:
                        marm_state = teleop.get_marm_joint_state()
                        _servo_ur5e(robot, marm_state[:6].tolist(), False)
                except RuntimeError as e:
                    logging.warning(f"[FREEDRIVE] servo command failed (transient): {e}")

                # Check for freedrive sub-mode request
                if freedrive_state.get("request_freedrive_sub"):
                    freedrive_state["request_freedrive_sub"] = False
                    logging.info("[FREEDRIVE] Switching back to freedrive (mArm stiff, follows UR5e)...")
                    # Stop servoJ and re-enter freedrive
                    enter_freedrive(robot, is_dual_arm)
                    teleop.set_marm_pid_profile("stiff")
                    freedrive_state["sub_mode"] = "freedrive"
                    logging.info("[FREEDRIVE] Freedrive active. 'i' for teleop, 'b' to exit.")

                time.sleep(0.02)  # ~50Hz

    except Exception as e:
        logging.error(f"[FREEDRIVE] Error in freedrive loop: {e}")

    # Exit freedrive
    freedrive_state["request_exit"] = False
    logging.info("====== [FREEDRIVE] Exiting, returning to start position... ======")

    # Restore mArm: disable torque, restore default PID
    try:
        teleop.set_marm_torque(False)
        teleop.set_marm_pid_profile("stiff")  # restore default
    except Exception:
        pass

    if freedrive_state["sub_mode"] == "freedrive":
        exit_freedrive_and_return(robot, is_dual_arm)
    else:
        # Was in teleop sub-mode, just stop servo and return
        if is_dual_arm:
            try:
                robot.left_arm._arm["rtde_c"].servoStop()
            except Exception:
                pass
            try:
                robot.right_arm._arm["rtde_c"].servoStop()
            except Exception:
                pass
            time.sleep(0.2)
            robot.left_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.left_arm.config.start_position).tolist(), 0.5, 0.5)
            robot.right_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.right_arm.config.start_position).tolist(), 0.5, 0.5)
        else:
            try:
                robot._arm["rtde_c"].servoStop()
            except Exception:
                pass
            time.sleep(0.2)
            robot._arm["rtde_c"].moveJ(np.deg2rad(robot.config.start_position).tolist(), 0.5, 0.5)

    freedrive_state["active"] = False
    freedrive_state["sub_mode"] = None
    logging.info("====== [FREEDRIVE] Resumed. Ready for teleop. ======")

    if dataset is not None:
        dataset.clear_episode_buffer()
    return True

def run_record(record_cfg: RecordConfig):
    try:
        dataset_name, data_version = generate_dataset_name(record_cfg)

        # Check joint offsets - this may update record_cfg.joint_offsets
        if not record_cfg.debug:
            if record_cfg.is_dual_arm:
                # Check both arms for dual-arm configuration
                check_joint_offsets(record_cfg, "left")
                check_joint_offsets(record_cfg, "right")
            else:
                check_joint_offsets(record_cfg)

        if record_cfg.is_dual_arm:
            # Create camera configurations for dual-arm setup
            left_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.left_wrist_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            left_exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.left_exterior_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            right_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.right_wrist_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            right_exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.right_exterior_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            # Create dual-arm teleop configuration
            teleop_config = UR5eTeleopConfig(
                port=record_cfg.port,
                control_mode=record_cfg.control_mode,
                left_arm=record_cfg.left_arm,
                right_arm=record_cfg.right_arm)

            # # Create dual-arm robot configurations
            # left_camera_config = {"wrist_image": left_wrist_cfg, "exterior_image": left_exterior_cfg}
            # right_camera_config = {"wrist_image": right_wrist_cfg, "exterior_image": right_exterior_cfg}

            # Only wrist camera is used for now
            left_camera_config = {"wrist_image": left_wrist_cfg}
            right_camera_config = {"wrist_image": right_wrist_cfg}

            left_robot_config = UR5eConfig(
                robot_ip=record_cfg.robot_left["ip"],
                gripper_port=record_cfg.robot_left["gripper_port"],
                cameras=left_camera_config,
                debug=record_cfg.debug,
                close_threshold=record_cfg.robot_left["close_threshold"],
                use_gripper=record_cfg.robot_left["use_gripper"],
                gripper_reverse=record_cfg.robot_left["gripper_reverse"],
                gripper_bin_threshold=record_cfg.robot_left["gripper_bin_threshold"],
                start_position=record_cfg.robot_left.get("start_position", [0, -30, 60, -100, 130, 0]))

            right_robot_config = UR5eConfig(
                robot_ip=record_cfg.robot_right["ip"],
                gripper_port=record_cfg.robot_right["gripper_port"],
                cameras=right_camera_config,
                debug=record_cfg.debug,
                close_threshold=record_cfg.robot_right["close_threshold"],
                use_gripper=record_cfg.robot_right["use_gripper"],
                gripper_reverse=record_cfg.robot_right["gripper_reverse"],
                gripper_bin_threshold=record_cfg.robot_right["gripper_bin_threshold"],
                start_position=record_cfg.robot_right.get("start_position", [0, -30, 60, -100, 130, 0]))

            # Initialize dual-arm robot and teleoperator
            robot = DualUR5e(left_robot_config, right_robot_config)
            teleop = UR5eTeleop(teleop_config)
        else:
            # Single-arm setup (backward compatibility)
            wrist_image_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.wrist_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            exterior_image_cfg = RealSenseCameraConfig(
                serial_number_or_name=record_cfg.exterior_cam_serial,
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION)

            # Note: record_cfg.joint_offsets may have been updated by check_joint_offsets
            camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}
            teleop_config = UR5eTeleopConfig(
                port=record_cfg.port,
                use_gripper=record_cfg.use_gripper,
                hardware_offsets=record_cfg.hardware_offsets,
                joint_ids=record_cfg.joint_ids,
                joint_offsets=record_cfg.joint_offsets,# This now uses the potentially updated value
                joint_signs=record_cfg.joint_signs,
                gripper_config=record_cfg.gripper_config,
                control_mode=record_cfg.control_mode)

            robot_config = UR5eConfig(
                robot_ip=record_cfg.robot_ip,
                gripper_port=record_cfg.gripper_port,
                cameras=camera_config,
                debug=record_cfg.debug,
                close_threshold=record_cfg.close_threshold,
                use_gripper=record_cfg.use_gripper,
                gripper_reverse=record_cfg.gripper_reverse,
                gripper_bin_threshold=record_cfg.gripper_bin_threshold,
                start_position=record_cfg.start_position)

            robot = UR5e(robot_config)
            teleop = UR5eTeleop(teleop_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_mera_period

        # Initialize the keyboard listener and rerun visualization
        _, events = init_keyboard_listener()
        init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        robot.connect()
        teleop.connect()

        # Set up key listener for 'p' (position snapshot), 'm' (freedrive), 'b' (back to start), 'i' (teleop sub-mode), 'o' (freedrive sub-mode)
        freedrive_state = {"active": False, "request_enter": False, "request_exit": False,
                           "sub_mode": None, "request_teleop": False, "request_freedrive_sub": False}
        p_listener = None
        try:
            from pynput import keyboard as _kb

            def _on_key_press(key):
                try:
                    if not hasattr(key, 'char') or key.char is None:
                        return
                    if key.char == 'p':
                        action = teleop.get_action()
                        obs = robot.get_observation()
                        pos_keys = [k for k in obs if k.endswith('.pos') or k.startswith('tcp_pose')]
                        print("\n====== [POSITION SNAPSHOT] ======")
                        print("  -- Teleop action --")
                        for k, v in action.items():
                            print(f"    {k}: {round(float(v), 6) if hasattr(v, '__float__') else v}")
                        print("  -- UR5e observation (pos/tcp) --")
                        for k in pos_keys:
                            v = obs[k]
                            print(f"    {k}: {round(float(v), 6) if hasattr(v, '__float__') else v}")
                        print("=================================\n")
                    elif key.char == 'm' and not freedrive_state["active"]:
                        print("\n[FREEDRIVE] 'm' pressed — requesting freedrive mode...")
                        freedrive_state["request_enter"] = True
                        events["exit_early"] = True
                    elif key.char == 'b' and freedrive_state["active"]:
                        print("\n[FREEDRIVE] 'b' pressed — exiting freedrive, returning to start...")
                        freedrive_state["request_exit"] = True
                    elif key.char == 'i' and freedrive_state["active"] and freedrive_state["sub_mode"] == "freedrive":
                        print("\n[FREEDRIVE] 'i' pressed — switching to teleop sub-mode...")
                        freedrive_state["request_teleop"] = True
                    elif key.char == 'o' and freedrive_state["active"] and freedrive_state["sub_mode"] == "teleop":
                        print("\n[FREEDRIVE] 'o' pressed — switching back to freedrive sub-mode...")
                        freedrive_state["request_freedrive_sub"] = True
                except Exception as e:
                    print(f"[key-listener] Error: {e}")

            p_listener = _kb.Listener(on_press=_on_key_press)
            p_listener.start()
            logging.info("Press 'p' for position snapshot, 'm' for freedrive mode, 'b' to exit freedrive, 'i' teleop sub-mode, 'o' freedrive sub-mode.")
        except Exception:
            logging.warning("Could not start key listener (headless or pynput unavailable).")

        episode_idx = 0

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            record_loop(
                robot=robot,
                events=events,
                fps=record_cfg.fps,
                teleop=teleop,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=record_cfg.episode_time_sec,
                single_task=record_cfg.task_description,
                display_data=record_cfg.display,
            )

            # If freedrive was requested mid-episode, handle it and restart episode
            if handle_freedrive_if_requested(freedrive_state, events, robot, record_cfg.is_dual_arm, teleop, dataset):
                continue

            if events["rerecord_episode"]:
                logging.info("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or events["rerecord_episode"]):
                while True:
                    if freedrive_state["request_enter"]:
                        handle_freedrive_if_requested(freedrive_state, events, robot, record_cfg.is_dual_arm, teleop, None)
                        continue
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    user_input = input("====== [WAIT] Press Enter to reset the environment ======")
                    if user_input == "":
                        break
                    else:
                        logging.info("====== [WARNING] Please press only Enter to continue ======")

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            episode_idx += 1

        # Clean up
        logging.info("Stop recording")
        if p_listener is not None:
            p_listener.stop()
        robot.disconnect()
        teleop.disconnect()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(record_cfg)