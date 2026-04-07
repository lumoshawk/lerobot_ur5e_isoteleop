from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_ur5e")
@dataclass
class UR5eTeleopConfig(TeleoperatorConfig):
    port: str
    baudrate: int = 57600
    control_mode: str = "isoteleop"

    # Single-arm configuration (optional, for backward compatibility)
    use_gripper: Optional[bool] = None
    hardware_offsets: Optional[list[float]] = None
    joint_ids: Optional[list[int]] = None
    joint_offsets: Optional[list[float]] = None
    joint_signs: Optional[list[int]] = None
    gripper_config: Optional[tuple[int, float, float]] = None
    gcomp_enable: Optional[bool] = False

    # Dual-arm configuration (optional)
    left_arm: Optional[dict] = None
    right_arm: Optional[dict] = None
