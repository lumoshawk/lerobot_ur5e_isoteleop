"""
Dual arm robot class that manages both arms using a single shared Dynamixel driver.
This avoids port conflicts when both arms use the same serial port.
"""

import math
import logging
from typing import Dict, Optional, Sequence
import numpy as np
from .driver import DynamixelDriver, FakeDynamixelDriver

logger = logging.getLogger(__name__)


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

        # Start virtual trigger for each arm that has trigger_rest_rad in gripper_config
        self._triggers = []
        for arm_cfg in (left_arm_config, right_arm_config):
            self._start_trigger_from_gripper_config(arm_cfg)

        # Start gravity compensation for each arm that has gcomp_enable
        self._gcomp_threads = []
        for arm_cfg, arm_key in [(left_arm_config, 'left_arm'), (right_arm_config, 'right_arm')]:
            self._start_gcomp_from_arm_config(arm_cfg, arm_key)

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

    def _start_trigger_from_gripper_config(self, arm_config):
        """Start a trigger thread if gripper_config has a 4th element (trigger_rest_rad).
        Format: [id, min, max, trigger_rest_rad, trigger_sign, trigger_hz]
        """
        gripper_cfg = arm_config.get('gripper_config')
        if not gripper_cfg or len(gripper_cfg) < 4:
            return
        servo_id = int(gripper_cfg[0])
        rest_deg = math.degrees(gripper_cfg[3])
        sign = int(gripper_cfg[4]) if len(gripper_cfg) >= 5 else 1
        loop_hz = int(gripper_cfg[5]) if len(gripper_cfg) >= 6 else 10
        try:
            from .virtual_trigger import TriggerThread
            trigger = TriggerThread(
                driver=self._driver,
                servo_id=servo_id,
                rest_deg=rest_deg,
                sign=sign,
                loop_hz=loop_hz,
            )
            trigger.start()
            self._triggers.append(trigger)
        except Exception as e:
            logger.warning(f"[TRIGGER] Failed to start trigger on servo {servo_id}: {e}")

    def stop_trigger(self):
        """Stop all trigger threads."""
        for trigger in self._triggers:
            trigger.stop()
        self._triggers.clear()

    def _start_gcomp_from_arm_config(self, arm_config, arm_key):
        """Start gravity compensation thread if arm_config has gcomp_enable."""
        if not arm_config.get('gcomp_enable', False):
            return
        try:
            from .gravity_compensation import GravityCompensationThread
            gcomp = GravityCompensationThread(
                driver=self._driver,
                arm_key=arm_key,
                joint_ids=arm_config['joint_ids'],
                loop_hz=10
            )
            gcomp.start()
            self._gcomp_threads.append(gcomp)
        except Exception as e:
            logger.warning(f"[GCOMP] Failed to start on {arm_key}: {e}")

    def start_gcomp(self):
        """Start gravity compensation threads (stops any existing ones first)."""
        self.stop_gcomp()
        for arm_cfg, arm_key in [(self.left_config, 'left_arm'), (self.right_config, 'right_arm')]:
            self._start_gcomp_from_arm_config(arm_cfg, arm_key)

    def stop_gcomp(self):
        """Stop all gravity compensation threads."""
        for gcomp in self._gcomp_threads:
            gcomp.stop()
        self._gcomp_threads.clear()

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
        self._last_pos = None
        if config.get('gcomp_enable', False):
            self._alpha = 1   # EMA smoothing factor (0=infinite smooth, 1=no smooth)
        else:
            self._alpha = 0.9

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

        # EMA smoothing to reduce Dynamixel jitter
        if self._last_pos is None:
            self._last_pos = pos.copy()
        else:
            pos = self._alpha * pos + (1 - self._alpha) * self._last_pos
            self._last_pos = pos.copy()

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