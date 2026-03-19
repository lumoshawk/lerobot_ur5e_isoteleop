"""
Dual arm robot class that manages both arms using a single shared Dynamixel driver.
This avoids port conflicts when both arms use the same serial port.
"""

from typing import Dict, Optional, Sequence
import numpy as np
from .driver import DynamixelDriver, FakeDynamixelDriver, PID_STIFF, PID_SOFT


class DualArmDynamixelRobot:
    """A robot class that manages both left and right arms using a single driver."""

    def __init__(
        self,
        left_arm_config: Dict,
        right_arm_config: Dict,
        port: str,
        real: bool = False,
        baudrate: int = 57600,
    ):
        """Initialize dual arm robot with shared driver.

        Args:
            left_arm_config: Configuration dictionary for left arm
            right_arm_config: Configuration dictionary for right arm
            port: Serial port for the Dynamixel connection
            real: Whether to use real hardware or fake driver
            baudrate: Communication baudrate
        """
        # Store configurations
        self.left_config = left_arm_config
        self.right_config = right_arm_config

        # Combine all joint IDs from both arms
        left_ids = list(left_arm_config['joint_ids'])
        right_ids = list(right_arm_config['joint_ids'])

        # Add gripper IDs if present
        if left_arm_config.get('use_gripper') and left_arm_config.get('gripper_config'):
            left_ids.append(left_arm_config['gripper_config'][0])
        if right_arm_config.get('use_gripper') and right_arm_config.get('gripper_config'):
            right_ids.append(right_arm_config['gripper_config'][0])

        self.all_ids = left_ids + right_ids
        self.left_ids = left_ids
        self.right_ids = right_ids

        # Combine hardware offsets for both arms (needed for get_positions)
        self.all_hardware_offsets = list(left_arm_config['hardware_offsets']) + list(right_arm_config['hardware_offsets'])

        # Create single shared driver for all servos
        if real:
            self._driver = DynamixelDriver(self.all_ids, port=port, baudrate=baudrate)
        else:
            self._driver = FakeDynamixelDriver(self.all_ids)

        self._torque_on = False

        # Create wrapper objects for each arm
        self._left_wrapper = ArmWrapper(
            self._driver,
            left_ids,
            left_arm_config,
            0  # Start index for left arm
        )
        self._right_wrapper = ArmWrapper(
            self._driver,
            right_ids,
            right_arm_config,
            len(left_ids)  # Start index for right arm
        )

    def get_left_arm_wrapper(self):
        """Return wrapper object for left arm."""
        return self._left_wrapper

    def get_right_arm_wrapper(self):
        """Return wrapper object for right arm."""
        return self._right_wrapper

    def set_torque_mode(self, mode: bool):
        """Enable/disable torque for all servos."""
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode
        self._left_wrapper._torque_on = mode
        self._right_wrapper._torque_on = mode


class ArmWrapper:
    """Wrapper class that provides DynamixelRobot-like interface for a single arm."""

    def __init__(
        self,
        driver,
        joint_ids: Sequence[int],
        config: Dict,
        start_index: int
    ):
        """Initialize arm wrapper.

        Args:
            driver: Shared DynamixelDriver instance
            joint_ids: IDs for this arm's servos
            config: Arm configuration dictionary
            start_index: Starting index in the shared driver's joint array
        """
        self._driver = driver
        self._joint_ids = joint_ids
        self._start_index = start_index
        self._end_index = start_index + len(joint_ids)

        # Store configuration
        self._hardware_offsets = config['hardware_offsets']
        self._joint_offsets = np.array(config['joint_offsets'])
        self._joint_signs = np.array(config['joint_signs'])
        self._use_gripper = config.get('use_gripper', False)

        # Handle gripper configuration
        self.gripper_open_close = None
        if self._use_gripper and config.get('gripper_config'):
            gripper_cfg = config['gripper_config']
            # Extend arrays for gripper
            self._joint_offsets = np.append(self._joint_offsets, 0.0)
            self._joint_signs = np.append(self._joint_signs, 1)
            self.gripper_open_close = (gripper_cfg[1], gripper_cfg[2])

        self._torque_on = False

    def num_dofs(self) -> int:
        """Return number of degrees of freedom."""
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        """Get joint positions for this arm."""
        # Get raw positions from the driver
        all_positions = self._driver.get_joints()

        # Apply hardware offsets (convert to degrees, add offsets, convert back)
        positions_deg = np.degrees(all_positions)

        # Apply hardware offsets only to the arm's joints (not grippers)
        for i in range(min(6, len(self._joint_ids))):
            positions_deg[self._start_index + i] += self._hardware_offsets[i]

        all_positions_with_offsets = np.radians(positions_deg)

        # Extract positions for this arm
        arm_positions = all_positions_with_offsets[self._start_index:self._end_index]

        # Apply joint offsets and signs
        pos = (arm_positions - self._joint_offsets) * self._joint_signs

        # Handle gripper mapping
        if self.gripper_open_close is not None:
            # Map gripper position to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command joint positions for this arm."""
        # Get current positions for all servos
        all_positions = self._driver.get_joints()

        # Update positions for this arm
        arm_positions = (joint_state / self._joint_signs + self._joint_offsets)
        all_positions[self._start_index:self._end_index] = arm_positions

        # Send updated positions to all servos
        self._driver.set_joints(all_positions.tolist())

    def set_torque_mode(self, mode: bool):
        """Set torque mode (handled by parent DualArmDynamixelRobot)."""
        # This is handled at the parent level since we need to control all servos together
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Return the current arm observations."""
        obs_dict = {}
        joint_state = self.get_joint_state()

        # Add joint positions
        for i in range(len(joint_state[:6])):
            obs_dict[f"joint_{i+1}.pos"] = joint_state[i]

        # Add gripper position if applicable
        gripper_pos = joint_state[-1] if self._use_gripper else None

        return {
            **obs_dict,
            "gripper_position": gripper_pos,
        }

    def set_pid_profile(self, profile: str):
        """Switch PID profile for this arm's servos only. profile: 'stiff' or 'soft'.
        PID gain registers (addr 80-84) are in RAM, writable with torque on.
        """
        gains = PID_STIFF if profile == "stiff" else PID_SOFT
        self._driver.set_pid_gains_for_ids(self._joint_ids, gains["p"], gains["i"], gains["d"])

    def command_from_ur5e_pos(self, ur5e_joints_rad: np.ndarray, gripper_pos: float = None):
        """Command this arm to match UR5e joint positions (reverse offset transform).

        Forward (get_joint_state):
          raw_deg[arm] += hardware_offsets  ->  raw_with_hw (radians)
          ur5e = (raw_with_hw[arm] - joint_offsets) * joint_signs

        Reverse:
          raw_with_hw = ur5e / joint_signs + joint_offsets
          raw_servo_deg = degrees(raw_with_hw) - hardware_offsets
          raw_servo = radians(raw_servo_deg)
        """
        # Step 1: reverse signs and joint_offsets
        raw_with_hw = ur5e_joints_rad / self._joint_signs[:6] + self._joint_offsets[:6]
        # Step 2: remove hardware offsets (added in degrees in get_joint_state)
        raw_deg = np.degrees(raw_with_hw)
        raw_deg -= np.array(self._hardware_offsets[:6])
        raw = np.radians(raw_deg)

        # Get current positions for all servos
        all_positions = self._driver.get_joints()

        # Update positions for this arm's joints
        for i in range(6):
            all_positions[self._start_index + i] = raw[i]

        # Handle gripper
        if gripper_pos is not None and self.gripper_open_close is not None:
            g_raw = gripper_pos * (self.gripper_open_close[1] - self.gripper_open_close[0]) + self.gripper_open_close[0]
            gripper_idx = self._start_index + 6  # gripper is after the 6 joints
            all_positions[gripper_idx] = g_raw

        self._driver.set_joints(all_positions.tolist())