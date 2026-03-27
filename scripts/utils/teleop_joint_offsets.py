#!/usr/bin/env python3
"""
Teleoperation Joint Offsets Calibration Tool

This script calibrates the hardware and joint offsets between the master (Dynamixel)
and slave (UR5e) robot arms for teleoperation.

Calibration Process:
1. Hardware Offsets: Calibrate the physical mounting differences
2. Joint Offsets: Calibrate the software offsets for accurate tracking
3. Save Configuration: Optionally save the calibration to config file
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import numpy as np
import logging
import time
import shutil
from datetime import datetime

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from lerobot_teleoperator_ur5e.dynamixel import DynamixelDriver
np.set_printoptions(suppress=True)

# ------------------------ Logging Setup ------------------------ #
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ------------------------ Constants ------------------------ #
# Predefined calibration position for hardware offset calibration (in degrees)
CALIBRATION_POSITION_DEG = [-90, -90, 0, -90, 0, 0]
CALIBRATION_POSITION_RAD = np.deg2rad(CALIBRATION_POSITION_DEG)

# Number of samples to take for hardware calibration
NUM_HARDWARE_SAMPLES = 3
SAMPLE_DELAY_SEC = 2.0

# ------------------------ Robot Functions ------------------------ #
def connect_to_ur5e(robot_ip: str) -> Optional[RTDEReceiveInterface]:
    """Connects to the UR5e robot and returns the RTDE interface."""
    try:
        logger.info("\n" + "="*60)
        logger.info("CONNECTING TO UR5e ROBOT")
        logger.info("="*60)
        logger.info(f"Robot IP: {robot_ip}")

        rtde_r = RTDEReceiveInterface(robot_ip)
        logger.info("✓ Successfully connected to UR5e robot")
        return rtde_r
    except Exception as e:
        logger.error("✗ Failed to connect to UR5e robot")
        logger.error(f"  Error: {e}")
        return None

def get_ur5e_joints(rtde_r: RTDEReceiveInterface) -> np.ndarray:
    """Get current joint positions from UR5e robot in radians."""
    return np.array(rtde_r.getActualQ())

# ------------------------ Dynamixel Functions ------------------------ #
def connect_to_dynamixel(cfg) -> Optional[DynamixelDriver]:
    """Connect to Dynamixel master arm."""
    try:
        logger.info("\n" + "="*60)
        logger.info("CONNECTING TO DYNAMIXEL MASTER ARM")
        logger.info("="*60)
        logger.info(f"Port: {cfg.port}")
        logger.info(f"Joint IDs: {cfg.joint_ids}")

        driver = DynamixelDriver(cfg.joint_ids, port=cfg.port, baudrate=57600)

        # Warmup reads
        for _ in range(10):
            driver.get_joints()
            time.sleep(0.01)

        logger.info("✓ Successfully connected to Dynamixel master arm")
        return driver
    except Exception as e:
        logger.error("✗ Failed to connect to Dynamixel master arm")
        logger.error(f"  Error: {e}")
        return None

def get_dynamixel_joints_with_offsets(driver: DynamixelDriver, hardware_offsets: List[float]) -> np.ndarray:
    """Get Dynamixel joint positions with hardware offsets applied."""
    positions_rad = driver.get_joints()
    positions_deg = np.degrees(positions_rad)
    positions_deg[:6] += hardware_offsets
    return np.radians(positions_deg)

# ------------------------ Calibration Functions ------------------------ #
def calibrate_hardware_offsets(driver: DynamixelDriver, rtde_r:RTDEReceiveInterface, cfg) -> Tuple[List[float], bool]:
    """
    Calibrate hardware offsets by having user position master arm at predefined position.
    Takes multiple samples and computes the mean.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: HARDWARE OFFSET CALIBRATION")
    logger.info("="*60)
    logger.info("\nThis step calibrates the physical mounting differences between")
    logger.info("the master and slave arms.")

    # Check current master arm position with existing hardware offsets
    if cfg.hardware_offsets and len(cfg.hardware_offsets) > 0:
        logger.info("\n" + "-"*60)
        logger.info("CHECKING CURRENT POSITIONS")
        logger.info("-"*60)

        # Get current master position with hardware offsets applied
        current_master_rad = driver.get_joints()
        current_master_deg = np.degrees(current_master_rad)
        master_with_hw_offsets = current_master_deg + np.array(cfg.hardware_offsets[:len(current_master_deg)])

        # Check against both saved calibration position and current UR5e position
        ur_current = get_ur5e_joints(rtde_r)
        ur_current_deg = np.degrees(ur_current)

        # Check if master matches saved calibration position (modulo 2π)
        matches_calibration = True
        matches_ur5e = True

        for i in range(len(current_master_deg)):
            # Check against calibration position
            diff_cal = ((master_with_hw_offsets[i] - CALIBRATION_POSITION_DEG[i] + 180) % 360) - 180
            if abs(diff_cal) > 10.0:  # 5 degree tolerance
                diff_mod = abs(diff_cal) % 360
                if min(diff_mod, 360 - diff_mod) > 10.0:
                    matches_calibration = False
                    logger.info(f"Joint {i+1} calibration mismatch: {diff_cal:.2f}° (mod {diff_mod:.2f}°)")

            # Check against current UR5e position
            diff_ur = ((master_with_hw_offsets[i] - ur_current_deg[i] + 180) % 360) - 180
            if abs(diff_ur) > 10.0:
                diff_mod = abs(diff_ur) % 360
                if min(diff_mod, 360 - diff_mod) > 10.0:
                    matches_ur5e = False
                    logger.info(f"Joint {i+1} UR5e mismatch: {diff_ur:.2f}° (mod {diff_mod:.2f}°)")

        if matches_calibration or matches_ur5e:
            logger.info("\n✓ Master arm appears to be already aligned!")
            logger.info("\nMaster position (with current hardware offsets):")
            for i in range(len(master_with_hw_offsets)):
                logger.info(f"  Joint {i+1}: {master_with_hw_offsets[i]:7.2f}°")

            if matches_ur5e:
                logger.info("\n✓ Matches current UR5e position:")
                for i in range(len(ur_current_deg)):
                    logger.info(f"  Joint {i+1}: {ur_current_deg[i]:7.2f}°")

            if matches_calibration:
                logger.info("\n✓ Matches saved calibration position:")
                for i in range(len(CALIBRATION_POSITION_DEG)):
                    logger.info(f"  Joint {i+1}: {CALIBRATION_POSITION_DEG[i]:7.2f}°")

            logger.info("\nCurrent hardware offsets:")
            for i, offset in enumerate(cfg.hardware_offsets[:len(current_master_deg)]):
                logger.info(f"  Joint {i+1}: {offset:7.2f}°")

            response = input("\n▶ Master arm is already aligned. Skip hardware calibration? (y/n): ").lower()
            if response == 'y':
                logger.info("\n✓ Skipping hardware offset calibration, using existing values.")
                return cfg.hardware_offsets, True
        else:
            logger.info("\n✗ Master arm hardware_offsets is not aligned. Proceeding with hardware calibration.")

    logger.info("\n" + "-"*60)

    # Ask user for UR5e position source
    logger.info("\n" + "-" * 60)
    logger.info("UR5e POSITION SOURCE OPTIONS:")
    logger.info("-" * 60)
    logger.info("1. Use CURRENT position from UR5e robot")
    logger.info("2. Use SAVED position ")
    logger.info("-" * 60)

    while True:
        choice = input("\n▶ Select option (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        logger.warning("Invalid choice. Please enter 1 or 2.")

    if choice == '1':
        # Get current UR5e position
        ur_joints = get_ur5e_joints(rtde_r)
        ur_joints_deg = np.degrees(ur_joints)
        logger.info("\nUsing CURRENT UR5e position")
    else:
        # Use SAVED position
        ur_joints_deg = CALIBRATION_POSITION_DEG
        ur_joints = np.radians(ur_joints_deg)
        logger.info("\nUse SAVED position")

    logger.info("CALIBRATION POSITION (degrees):")
    logger.info("-"*60)
    for i, angle in enumerate(ur_joints_deg):
        logger.info(f"  Joint {i+1}: {angle:6.1f}°")
    logger.info("-"*60)

    logger.info("\n⚠ IMPORTANT INSTRUCTIONS:")
    logger.info("1. Move the UR5e robot to the calibration position shown above")
    logger.info("2. You will now calibrate each joint of the master arm one by one.")
    logger.info("3. For each joint, manually align it to match EXACTLY the UR5e position and press ENTER.")
    logger.info("4. The system will take 3 samples per joint.")

    num_joints = len(ur_joints_deg)
    mean_master_deg = np.full(num_joints, np.nan)
    std_master_deg = np.full(num_joints, np.nan)

    existing_offsets = list(cfg.hardware_offsets[:num_joints]) if cfg.hardware_offsets else []
    if len(existing_offsets) < num_joints:
        existing_offsets.extend([0.0] * (num_joints - len(existing_offsets)))
    hardware_offsets = np.array(existing_offsets, dtype=float)

    logger.info("\nSelect hardware offset calibration mode:")
    logger.info("1. Calibrate ALL joints")
    logger.info("2. Calibrate ONE joint")
    logger.info("3. Calibrate MULTIPLE joints")
    while True:
        mode = input("\n▶ Select option (1, 2, or 3): ").strip()
        if mode in ['1', '2', '3']:
            break
        logger.warning("Invalid choice. Please enter 1, 2, or 3.")

    if mode == '1':
        joints_to_calibrate = list(range(num_joints))
    elif mode == '2':
        while True:
            choice = input(f"▶ Enter joint number to calibrate (1-{num_joints}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= num_joints:
                joints_to_calibrate = [int(choice) - 1]
                break
            logger.warning(f"Invalid joint number. Please enter 1-{num_joints}.")
    else:
        while True:
            choice = input(f"▶ Enter joint numbers separated by commas (1-{num_joints}): ").strip()
            try:
                joints = sorted({int(x.strip()) - 1 for x in choice.split(',') if x.strip()})
            except ValueError:
                joints = []

            if joints and all(0 <= j < num_joints for j in joints):
                joints_to_calibrate = joints
                break
            logger.warning(f"Invalid input. Example: 1,3,5 (range 1-{num_joints}).")

    for joint_idx in joints_to_calibrate:
        logger.info(f"\n" + "-"*40)
        logger.info(f"CALIBRATING JOINT {joint_idx + 1}")
        logger.info(f"Target angle: {ur_joints_deg[joint_idx]:6.1f}°")
        logger.info("-" * 40)
        input(f"▶ Align Joint {joint_idx + 1} and press ENTER...")
        time.sleep(1) # Short delay to let arm settle
        
        # Collect multiple samples for this joint
        logger.info(f"Collecting {NUM_HARDWARE_SAMPLES} samples for Joint {joint_idx + 1}...")
        joint_samples = []
        
        for i in range(NUM_HARDWARE_SAMPLES):
            print(f"  Sample {i+1}/{NUM_HARDWARE_SAMPLES}...", end='', flush=True)
            master_joints_rad = driver.get_joints()
            master_joints_deg = np.degrees(master_joints_rad)
            joint_samples.append(master_joints_deg[joint_idx])
            print(f" ✓ [{master_joints_deg[joint_idx]:.2f}°]")
            
            if i < NUM_HARDWARE_SAMPLES - 1:
                time.sleep(SAMPLE_DELAY_SEC)
                
        # Calculate mean for this joint
        samples_array = np.array(joint_samples)
        mean_master_deg[joint_idx] = np.mean(samples_array)
        std_master_deg[joint_idx] = np.std(samples_array)
        hardware_offsets[joint_idx] = ur_joints_deg[joint_idx] - mean_master_deg[joint_idx]

    unchanged_joints = [i for i in range(num_joints) if i not in joints_to_calibrate]
    if unchanged_joints:
        logger.info("\nKeeping existing hardware offsets for joints not calibrated in this run:")
        for idx in unchanged_joints:
            logger.info(f"  Joint {idx+1}: {hardware_offsets[idx]:7.2f}° (unchanged)")

    # Display results
    logger.info("\n" + "-"*60)
    logger.info("CALIBRATION RESULTS")
    logger.info("-"*60)
    logger.info("\nMaster Arm Readings (mean ± std):")
    for i in range(num_joints):
        if np.isnan(mean_master_deg[i]):
            logger.info(f"  Joint {i+1}: {'N/A':>7} ± {'N/A':>5} (not calibrated)")
        else:
            logger.info(f"  Joint {i+1}: {mean_master_deg[i]:7.2f}° ± {std_master_deg[i]:5.2f}°")

    logger.info("\nCalculated Hardware Offsets:")
    for i, offset in enumerate(hardware_offsets):
        logger.info(f"  Joint {i+1}: {offset:7.2f}°")

    # Check if standard deviation is too high (potential measurement error)
    calibrated_std = std_master_deg[~np.isnan(std_master_deg)]
    max_std = np.max(calibrated_std) if len(calibrated_std) else 0.0
    if max_std > 2.0:
        logger.warning(f"\n⚠ Warning: High standard deviation detected (max: {max_std:.2f}°)")
        logger.warning("  This might indicate the arm moved during sampling.")
        response = input("  Continue anyway? (y/n): ").lower()
        if response != 'y':
            return hardware_offsets.tolist(), False

    logger.info("\n✓ Hardware offset calibration complete!")
    return hardware_offsets.tolist(), True

def calibrate_joint_offsets(driver: DynamixelDriver, rtde_r: RTDEReceiveInterface,
                           hardware_offsets: List[float], cfg) -> Tuple[List[float], bool]:
    """
    Calibrate joint offsets using the current UR5e position and new hardware offsets.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: JOINT OFFSET CALIBRATION")
    logger.info("="*60)
    logger.info("\nThis step calibrates the software offsets to ensure accurate")
    logger.info("tracking between master and slave arms.")

    # Get current UR5e position
    ur_joints = get_ur5e_joints(rtde_r)
    ur_joints_deg = np.degrees(ur_joints)

    logger.info("\n" + "-"*60)
    logger.info("CURRENT UR5e POSITION (degrees):")
    logger.info("-"*60)
    for i, angle in enumerate(ur_joints_deg):
        logger.info(f"  Joint {i+1}: {angle:7.2f}°")

    logger.info("\n⚠ IMPORTANT INSTRUCTIONS:")
    logger.info("1. Move the UR5e robot to start position")
    logger.info("2. Move the master arm to match the UR5e position as closely as possible")
    logger.info("3. The system will calculate offsets to align the arms")

    input("\n▶ Press ENTER when the master arm is in position...")

    # Get master arm position with hardware offsets
    master_joints_with_hw = get_dynamixel_joints_with_offsets(driver, hardware_offsets)
    master_joints_with_hw_deg = np.degrees(master_joints_with_hw)

    logger.info("\nMaster Arm Position (with hardware offsets):")
    for i, angle in enumerate(master_joints_with_hw_deg):
        logger.info(f"  Joint {i+1}: {angle:7.2f}°")

    # Calculate joint offsets
    joint_offsets = []
    errors = []

    for i in range(len(cfg.joint_ids)):
        joint_sign = cfg.joint_signs[i]

        # Search for best offset
        best_offset = 0
        best_error = float('inf')

        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 33):  # intervals of pi/2
            joint_val = joint_sign * (master_joints_with_hw[i] - offset)
            error = np.abs(joint_val - ur_joints[i])
            if error < best_error:
                best_error = error
                best_offset = offset

        joint_offsets.append(best_offset)
        errors.append(np.degrees(best_error))

    # Display results
    logger.info("\n" + "-"*60)
    logger.info("CALIBRATION RESULTS")
    logger.info("-"*60)
    logger.info("\nCalculated Joint Offsets (radians):")
    for i, offset in enumerate(joint_offsets):
        logger.info(f"  Joint {i+1}: {offset:7.3f} rad ({np.degrees(offset):7.2f}°)")

    logger.info("\nAlignment Errors (after calibration):")
    for i, error in enumerate(errors):
        status = "✓" if error < 5.0 else "⚠"
        logger.info(f"  Joint {i+1}: {error:5.2f}° {status}")

    # Check if errors are acceptable
    max_error = max(errors)
    if max_error > 10.0:
        logger.warning(f"\n⚠ Warning: Large alignment error detected (max: {max_error:.2f}°)")
        logger.warning("  The master arm might not be positioned correctly.")
        response = input("  Continue anyway? (y/n): ").lower()
        if response != 'y':
            return [round(x, 3) for x in joint_offsets], False

    logger.info("\n✓ Joint offset calibration complete!")
    return [round(x, 3) for x in joint_offsets], True

def save_calibration(cfg_path: Path, hardware_offsets: List[float],
                     joint_offsets: List[float], arm_name: str = None) -> bool:
    """Save calibration results to configuration file with backup."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: SAVE CONFIGURATION")
    logger.info("="*60)

    # Display what will be saved
    logger.info("\nCalibration values to save:")
    logger.info("\nHardware Offsets:")
    for i, offset in enumerate(hardware_offsets):
        logger.info(f"  Joint {i+1}: {offset:7.2f}")

    logger.info("\nJoint Offsets:")
    for i, offset in enumerate(joint_offsets):
        logger.info(f"  Joint {i+1}: {offset:7.3f}")

    response = input("\n▶ Save these calibration values to cfg.yaml? (y/n): ").lower()

    if response != 'y':
        logger.info("✗ Calibration not saved.")
        return False

    try:
        # Create backup
        backup_path = cfg_path.with_suffix(f'.yaml.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        shutil.copy2(cfg_path, backup_path)
        logger.info(f"\n✓ Created backup: {backup_path.name}")

        # Read the original file as text to preserve formatting
        with open(cfg_path, 'r') as f:
            lines = f.readlines()

        # Find and update hardware_offsets line
        in_hardware_offsets = False
        in_joint_offsets = False
        updated_lines = []
        hardware_offset_indent = ""
        joint_offset_indent = ""

        i = 0
        while i < len(lines):
            line = lines[i]

            # For dual-arm mode, we need to check for specific arm sections
            if arm_name:
                # Check for hardware_offsets in the specific arm section
                if 'hardware_offsets:' in line and arm_name in ''.join(lines[max(0, i-5):i]):
                    updated_lines.append(line)
                    # Get the indent level
                    hardware_offset_indent = line[:len(line) - len(line.lstrip())]
                    # Format as in original: list on same line
                    hw_offsets_str = f"{hardware_offset_indent}hardware_offsets: [{', '.join([str(float(o)) for o in hardware_offsets])}] # The Calibration offsets between the master and the slave robot\n"
                    updated_lines[-1] = hw_offsets_str

                    # Skip old hardware_offsets lines if they were on multiple lines
                    i += 1
                    while i < len(lines) and (lines[i].strip().startswith('-') or (lines[i].strip() and not ':' in lines[i])):
                        i += 1
                    continue

                # Check for joint_offsets in the specific arm section
                elif 'joint_offsets:' in line and arm_name in ''.join(lines[max(0, i-10):i]):
                    updated_lines.append(line)
                    # Get the indent level
                    joint_offset_indent = line[:len(line) - len(line.lstrip())]
                    # Format as in original: list on same line
                    joint_offsets_str = f"{joint_offset_indent}joint_offsets: [{', '.join([str(float(o)) for o in joint_offsets])}]\n"
                    updated_lines[-1] = joint_offsets_str

                    # Skip old joint_offsets lines if they were on multiple lines
                    i += 1
                    while i < len(lines) and (lines[i].strip().startswith('-') or
                                             '!!python' in lines[i] or
                                             'numpy' in lines[i] or
                                             '*id' in lines[i] or
                                             '!!binary' in lines[i] or
                                             (lines[i].strip() and not ':' in lines[i] and not lines[i].strip().startswith('#'))):
                        i += 1
                    continue
                else:
                    updated_lines.append(line)
                    i += 1
            else:
                # Single-arm mode (original logic)
                # Check for hardware_offsets
                if 'hardware_offsets:' in line and 'dynamixel_config' in ''.join(lines[max(0, i-5):i]):
                    updated_lines.append(line)
                    # Get the indent level
                    hardware_offset_indent = line[:len(line) - len(line.lstrip())]
                    # Format as in original: list on same line
                    hw_offsets_str = f"{hardware_offset_indent}hardware_offsets: [{', '.join([str(float(o)) for o in hardware_offsets])}] # The Calibration offsets between the master and the slave robot\n"
                    updated_lines[-1] = hw_offsets_str

                    # Skip old hardware_offsets lines if they were on multiple lines
                    i += 1
                    while i < len(lines) and (lines[i].strip().startswith('-') or (lines[i].strip() and not ':' in lines[i])):
                        i += 1
                    continue

                # Check for joint_offsets
                elif 'joint_offsets:' in line and 'dynamixel_config' in ''.join(lines[max(0, i-10):i]):
                    updated_lines.append(line)
                    # Get the indent level
                    joint_offset_indent = line[:len(line) - len(line.lstrip())]
                    # Format as in original: list on same line
                    joint_offsets_str = f"{joint_offset_indent}joint_offsets: [{', '.join([str(float(o)) for o in joint_offsets])}]\n"
                    updated_lines[-1] = joint_offsets_str

                    # Skip old joint_offsets lines if they were on multiple lines
                    i += 1
                    while i < len(lines) and (lines[i].strip().startswith('-') or
                                             '!!python' in lines[i] or
                                             'numpy' in lines[i] or
                                             '*id' in lines[i] or
                                             '!!binary' in lines[i] or
                                             (lines[i].strip() and not ':' in lines[i] and not lines[i].strip().startswith('#'))):
                        i += 1
                    continue
                else:
                    updated_lines.append(line)
                    i += 1

        # Write the updated config
        with open(cfg_path, 'w') as f:
            f.writelines(updated_lines)

        logger.info(f"✓ Saved calibration to: {cfg_path}")
        logger.info("\n" + "="*60)
        logger.info("CALIBRATION COMPLETE!")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"✗ Failed to save calibration: {e}")
        return False

# ------------------------ Config Loader ------------------------ #
class CalibrationConfig:
    """Configuration for calibration."""
    def __init__(self, cfg: Dict[str, Any]):
        teleop = cfg["teleop"]
        robot = cfg["robot"]

        # Check if dual-arm mode is enabled
        self.is_dual_arm = cfg.get("dual_arm_mode", False)

        if self.is_dual_arm:
            # For dual-arm calibration, ask which arm to calibrate
            print("\n" + "="*60)
            print("DUAL-ARM MODE DETECTED")
            print("="*60)
            print("Which arm would you like to calibrate?")
            print("1. Left arm")
            print("2. Right arm")

            while True:
                choice = input("\nSelect option (1 or 2): ").strip()
                if choice in ['1', '2']:
                    break
                print("Invalid choice. Please enter 1 or 2.")

            if choice == '1':
                arm_cfg = teleop["left_arm"]
                robot_cfg = robot["left_arm"]
                print("\n✓ Calibrating LEFT arm")
            else:
                arm_cfg = teleop["right_arm"]
                robot_cfg = robot["right_arm"]
                print("\n✓ Calibrating RIGHT arm")

            self.arm_name = "left_arm" if choice == '1' else "right_arm"
            self.port = teleop["port"]
            self.joint_ids = arm_cfg["joint_ids"]
            self.joint_signs = arm_cfg["joint_signs"]
            self.hardware_offsets = arm_cfg["hardware_offsets"]
            self.joint_offsets = arm_cfg["joint_offsets"]
            self.robot_ip = robot_cfg["ip"]
        else:
            # Single-arm config
            dxl_cfg = teleop["dynamixel_config"]

            # Teleop config
            self.port = dxl_cfg["port"]
            self.joint_ids = dxl_cfg["joint_ids"]
            self.joint_signs = dxl_cfg["joint_signs"]
            self.hardware_offsets = dxl_cfg["hardware_offsets"]
            self.joint_offsets = dxl_cfg["joint_offsets"]

            # Robot config
            self.robot_ip: str = robot["ip"]
            self.arm_name = None


#These are for run_record to call
def get_start_joints(cfg) -> List[float]:
    """Connects to the UR5e robot, moves to start position, and retrieves joint positions."""
    try:
        logger.info("\n===== [ROBOT] Connecting to UR5e robot =====")
        rtde_r = RTDEReceiveInterface(cfg.robot_ip)
        rtde_c = RTDEControlInterface(cfg.robot_ip)

        joint_positions = rtde_r.getActualQ()
        logger.info(f"[ROBOT] Current joint positions (rad): {joint_positions}")
        logger.info(f"[ROBOT] Current joint positions (deg): {np.rad2deg(joint_positions)}")
        logger.info("===== [ROBOT] UR5e connected successfully =====\n")

        start_position_deg = getattr(cfg, 'start_position', [0, -30, 60, -100, 130, 0])
        start_position_rad = np.deg2rad(start_position_deg)

        logger.info("===== [ROBOT] Moving to start position =====")
        logger.info(f"[ROBOT] Target position (deg): {start_position_deg}")
        logger.info(f"[ROBOT] Target position (rad): {[round(r, 4) for r in start_position_rad]}")

        # Move to start position with moveJ (joint space movement)
        rtde_c.moveJ(start_position_rad, speed=0.5, acceleration=0.5)

        # Get final position after movement

        final_position = rtde_r.getActualQ()
        formatted_final = [round(j, 4) for j in final_position]
        logger.info(f"[ROBOT] Final joint positions: {formatted_final}")
        logger.info("===== [ROBOT] Start position reached =====\n")

        # Disconnect control interface
        rtde_c.disconnect()

        return final_position
    except Exception as e:
        logger.error("===== [ERROR] Failed to connect to UR5e robot =====")
        logger.error(f"Exception: {e}\n")
        return []


# ------------------------ Offset Calculation ------------------------ #
def compute_joint_offsets(cfg, start_joints: List[float]):
    """Compute offsets for Dynamixel joints to match the UR5e joint positions."""

    driver = DynamixelDriver(cfg.joint_ids, port=cfg.port, baudrate=57600)

    # Warmup reads
    for _ in range(10):
        driver.get_joints()

    def joint_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        """Calculate error between adjusted joint state and start joint."""
        joint_sign = cfg.joint_signs[index]
        joint_val = joint_sign * (joint_state[index] - offset)
        return np.abs(joint_val - start_joints[index])

    # Compute best offsets
    curr_joints = driver.get_joints_deg()
    curr_modified_joints = driver.get_positions(cfg.hardware_offsets)
    logger.info("Dynamixel current joint positions: %s", curr_joints)
    logger.info("Dynamixel current modified joint positions (rad): %s", curr_modified_joints)
    logger.info("Dynamixel current modified joint positions (deg): %s", np.rad2deg(curr_modified_joints))

    # Close driver
    driver.close()

    best_offsets = []

    for i in range(len(cfg.joint_ids)):
        best_offset = 0
        best_error = float('inf')
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 33):  # intervals of pi/2
            error = joint_error(offset, i, curr_modified_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(float(best_offset))

    logger.info("Joint offsets: %s", [round(x, 3) for x in best_offsets])

    return [round(x, 3) for x in best_offsets]

# ------------------------ Config Loader ------------------------ #
class RecordConfig:
    """Configuration for teleoperation and robot."""
    def __init__(self, cfg: Dict[str, Any]):
        teleop = cfg["teleop"]
        robot = cfg["robot"]

        # Check if dual-arm mode is enabled from config
        self.is_dual_arm = cfg.get("dual_arm_mode", False)

        if self.is_dual_arm:
            # Dual-arm teleop config
            self.port = teleop["port"]
            self.left_arm = teleop["left_arm"]
            self.right_arm = teleop["right_arm"]

            # Dual-arm robot config
            self.robot_left = robot["left_arm"]
            self.robot_right = robot["right_arm"]

            # For dual-arm, we'll calibrate left arm by default
            # (can be extended to calibrate both arms)
            self.joint_ids = self.left_arm["joint_ids"]
            self.joint_signs = self.left_arm["joint_signs"]
            self.hardware_offsets = self.left_arm["hardware_offsets"]
            self.robot_ip = self.robot_left["ip"]
        else:
            # Single-arm config (backward compatibility)
            dxl_cfg = teleop["dynamixel_config"]
            self.port = dxl_cfg["port"]
            self.joint_ids = dxl_cfg["joint_ids"]
            self.joint_signs = dxl_cfg["joint_signs"]
            self.hardware_offsets = dxl_cfg["hardware_offsets"]
            self.robot_ip: str = robot["ip"]

def run(record_cfg):
    start_joints = get_start_joints(record_cfg)
    if start_joints:
        return compute_joint_offsets(record_cfg, start_joints)
    else:
        raise RuntimeError("Failed to retrieve start joints from UR5e robot.")

# ------------------------ Main ------------------------ #

# ------------------------ Main Function ------------------------ #
def main():
    """Main calibration routine."""
    logger.info("\n" + "="*60)
    logger.info(" TELEOPERATION JOINT OFFSETS CALIBRATION TOOL")
    logger.info("="*60)
    logger.info("\nThis tool calibrates the offsets between the master")
    logger.info("(Dynamixel) and slave (UR5e) robot arms.")

    # Load configuration
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"

    logger.info(f"\nConfiguration file: {cfg_path}")

    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return 1

    calibration_cfg = CalibrationConfig(cfg["record"])

    # Connect to robots
    rtde_r = connect_to_ur5e(calibration_cfg.robot_ip)
    if rtde_r is None:
        return 1

    driver = connect_to_dynamixel(calibration_cfg)
    if driver is None:
        return 1

    try:
        # Step 1: Calibrate hardware offsets
        hardware_offsets, success = calibrate_hardware_offsets(driver, rtde_r, calibration_cfg)
        if not success:
            logger.info("\n✗ Hardware calibration cancelled.")
            return 1

        # Step 2: Calibrate joint offsets with new hardware offsets
        joint_offsets, success = calibrate_joint_offsets(
            driver, rtde_r, hardware_offsets, calibration_cfg
        )
        if not success:
            logger.info("\n✗ Joint calibration cancelled.")
            return 1

        # Step 3: Save calibration
        save_calibration(cfg_path, hardware_offsets, joint_offsets, calibration_cfg.arm_name)

    except KeyboardInterrupt:
        logger.info("\n\n✗ Calibration interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        driver.close()
        logger.info("\n✓ Closed connections.")

    return 0

if __name__ == "__main__":
    exit(main())
