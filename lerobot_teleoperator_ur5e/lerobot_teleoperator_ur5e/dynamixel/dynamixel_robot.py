from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from .robot import Robot
from .driver import PID_STIFF, PID_SOFT


class DynamixelRobot(Robot):
    """A class representing a UR5e robot."""

    def __init__(
        self,
        port: str,
        hardware_offsets: Sequence[float],
        joint_ids: Sequence[int],
        joint_offsets: Sequence[float],
        joint_signs: Sequence[int],
        real: bool = False,
        baudrate: int = 57600,
        use_gripper: bool = True,
        gripper_config: Optional[Tuple[int, float, float]] = None,
    ):  
        from .driver import (
            DynamixelDriver,
            DynamixelDriverProtocol,
            FakeDynamixelDriver,
        )

        if joint_offsets is None or len(joint_offsets) != 6:
            raise ValueError(f"joint_offsets must be a sequence of length 6, got {joint_offsets}")

        if joint_signs is None or len(joint_signs) != 6:
            raise ValueError(f"joint_signs must be a sequence of length 6, got {joint_signs}")

        if joint_ids is None or len(joint_ids) != 6:
            raise ValueError(f"joint_ids must be a sequence of length 6, got {joint_ids}")


        self.gripper_open_close: Optional[Tuple[float, float]]

        if use_gripper and gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1],
                gripper_config[2],
            )
        else:
            self.gripper_open_close = None
        
        self._use_gripper = use_gripper
        self._hardware_offsets = hardware_offsets
        self._joint_ids = joint_ids
        self._joint_offsets = np.array(joint_offsets)
        self._joint_signs = np.array(joint_signs)
        self._driver: DynamixelDriverProtocol

        if real:
            self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
            # self._driver.set_torque_mode(False)
        else:
            self._driver = FakeDynamixelDriver(joint_ids)

        self._torque_on = False
        self._last_pos = None
        self._alpha = 1.0
        self.record_time = 0
        
    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_positions(self._hardware_offsets) - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()
        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )

            g_pos = min(max(0, g_pos), 1)
            # g_pos = 0.0 if g_pos < self._close_threshold else 1.0
            pos[-1] = g_pos
            
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.set_joints((joint_state / self._joint_signs + self._joint_offsets).tolist())

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return the current robot observations.

        If `self._use_gripper` is True, split the gripper state from joint_state.
        """
        obs_dict = {}
        joint_state = self.get_joint_state()
        for i in range(len(joint_state[:6])):
            obs_dict[f"joint_{i+1}.pos"] = joint_state[i]

        if not self._use_gripper:
            joint_state[-1] = None

        return {
            **obs_dict,
            "gripper_position": joint_state[-1],
        }

    def set_pid_profile(self, profile: str):
        """Switch PID profile. profile: 'stiff' or 'soft'.
        PID gain registers (addr 80-84) are in RAM, writable with torque on.
        """
        gains = PID_STIFF if profile == "stiff" else PID_SOFT
        self._driver.set_pid_gains(gains["p"], gains["i"], gains["d"])

    def command_from_ur5e_pos(self, ur5e_joints_rad: np.ndarray, gripper_pos: float = None):
        """Command mArm to match UR5e joint positions (reverse offset transform).
        ur5e_joints_rad: 6-element array of UR5e joint positions in radians.
        gripper_pos: optional gripper value in [0,1] range.

        Forward: get_joint_state does:
          raw_with_hw = get_positions(hw_offsets)  # raw_rad + hw_offsets (in deg then back to rad)
          ur5e = (raw_with_hw - joint_offsets) * joint_signs

        Reverse:
          raw_with_hw = ur5e / joint_signs + joint_offsets
          raw_servo_deg = degrees(raw_with_hw) - hardware_offsets
          raw_servo = radians(raw_servo_deg)
          set_joints(raw_servo)
        """
        # Step 1: reverse signs and joint_offsets
        raw_with_hw = ur5e_joints_rad / self._joint_signs[:6] + self._joint_offsets[:6]
        # Step 2: remove hardware offsets (they are added in degrees in get_positions)
        raw_deg = np.degrees(raw_with_hw)
        raw_deg -= np.array(self._hardware_offsets[:6])
        raw = np.radians(raw_deg)

        if gripper_pos is not None and self.gripper_open_close is not None:
            g_raw = gripper_pos * (self.gripper_open_close[1] - self.gripper_open_close[0]) + self.gripper_open_close[0]
            raw = np.append(raw, g_raw)
        elif self.gripper_open_close is not None:
            # Keep current gripper position — read raw gripper directly from driver
            all_joints = self._driver.get_joints()
            raw = np.append(raw, all_joints[6])  # gripper is the 7th servo
        self._driver.set_joints(raw.tolist())

