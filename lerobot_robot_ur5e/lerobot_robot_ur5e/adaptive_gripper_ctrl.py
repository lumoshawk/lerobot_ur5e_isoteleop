"""
Phase-dependent gripper controller for PGI140 (via pyDHgripper.PGE).

State machine:
  FREE    — master angle -> continuous gripper position, force = F_min
  CONTACT — master angle -> force setpoint (relative to contact point),
            position micro-adjusts. Releasing master decreases force,
            gripper opens, and read_state() != 2 exits contact.

Transitions:
  FREE -> CONTACT:  read_state() == 2  (gripper physically contacted object)
  CONTACT -> FREE:  read_state() != 2  (gripper physically lost contact)
"""

import time
import logging

logger = logging.getLogger(__name__)


class AdaptiveGripperController:

    def __init__(self, gripper, config):
        self._gripper = gripper
        self._cfg = config.adaptive_gripper
        self._close_threshold = config.close_threshold
        self._gripper_reverse = config.gripper_reverse
        self._gripper_bin_threshold = config.gripper_bin_threshold
        self._master_max = self._cfg.get("master_max", 1.0)

        # State
        self._master_pos = 0.0
        self._contact = False
        self._theta_contact = None     # remapped master angle at first contact
        self._pos_contact = None       # gripper position (0-1000) at first contact
        self._last_gripper_bin = 0.0
        self._gripper_pos_reading = None

        logger.info("[ADAPTIVE-GRIPPER] Initialized: f_min=%d, k_f=%d, master_max=%.3f",
                     self._cfg["f_min"], self._cfg["k_f"], self._master_max)

    def update_master(self, master_gripper_pos: float):
        self._master_pos = master_gripper_pos

    def _remap(self, raw):
        """Remap raw master_pos [0, master_max] -> [0, 1]."""
        if self._master_max > 0:
            return min(raw / self._master_max, 1.0)
        return raw

    def run_loop(self):
        self._gripper.set_force(self._cfg["f_min"])
        self._gripper.set_vel(self._cfg["vel_free"])
        last_p_cmd = -1
        last_f_set = self._cfg["f_min"]
        mode1_fv_set = True
        mode2_vel_set = False
        obs_counter = 0

        while True:
            theta_m = self._remap(self._master_pos)

            # --- Read gripper status every iteration (~80ms) ---
            try:
                status = self._gripper.read_state()
            except Exception:
                status = -1  # comm error, keep current state

            # --- State transitions (hardware-driven) ---
            if not self._contact and status == 2:
                # ENTER: gripper physically contacted object
                self._contact = True
                self._theta_contact = theta_m
                try:
                    self._pos_contact = self._gripper.read_pos()
                except Exception:
                    self._pos_contact = last_p_cmd if last_p_cmd >= 0 else 500
                mode2_vel_set = False
                mode1_fv_set = False
                logger.info("[ADAPTIVE-GRIPPER] CONTACT at theta=%.3f, pos=%d",
                            theta_m, self._pos_contact)

            elif self._contact and status != 2 and status >= 0:
                # EXIT: gripper physically lost contact (status changed, not comm error)
                self._contact = False
                self._theta_contact = None
                self._pos_contact = None
                mode1_fv_set = False
                mode2_vel_set = False
                logger.info("[ADAPTIVE-GRIPPER] RELEASED (status=%d, theta=%.3f)",
                            status, theta_m)

            # --- Control ---
            if not self._contact:
                # MODE 1: continuous position from master
                if not mode1_fv_set:
                    self._gripper.set_force(self._cfg["f_min"])
                    self._gripper.set_vel(self._cfg["vel_free"])
                    mode1_fv_set = True

                p_cmd = int(theta_m * 1000)
                if self._gripper_reverse:
                    p_cmd = 1000 - p_cmd
                p_cmd = max(0, min(1000, p_cmd))

                if abs(p_cmd - last_p_cmd) > 5:
                    self._gripper.set_pos(val=p_cmd, blocking=False)
                    last_p_cmd = p_cmd

                self._last_gripper_bin = 0.0 if theta_m < self._close_threshold else 1.0
            else:
                # MODE 2: force from master delta relative to contact point
                if not mode2_vel_set:
                    self._gripper.set_vel(self._cfg["vel_grasp"])
                    mode2_vel_set = True

                # delta > 0 = squeezing harder, delta < 0 = releasing
                delta = theta_m - self._theta_contact
                f_set = int(self._cfg["f_min"] + self._cfg["k_f"] * max(delta, 0))
                f_set = max(20, min(self._cfg["f_max"], f_set))

                if f_set != last_f_set:
                    self._gripper.set_force(f_set)
                    last_f_set = f_set

                # Position: micro-adjust from contact point
                # delta < 0 means opening back from contact position
                p_cmd = int(self._pos_contact + self._cfg["epsilon"] * delta * 1000)
                p_cmd = max(0, min(1000, p_cmd))

                if abs(p_cmd - last_p_cmd) > 2:
                    self._gripper.set_pos(val=p_cmd, blocking=False)
                    last_p_cmd = p_cmd

            # --- Observation (every 5th iteration) ---
            obs_counter += 1
            if obs_counter >= 5:
                obs_counter = 0
                try:
                    raw_pos = self._gripper.read_pos() / 1000.0
                    if self._gripper_reverse:
                        raw_pos = 1.0 - raw_pos
                    self._gripper_pos_reading = raw_pos
                except Exception:
                    pass

            time.sleep(0.01)

    @property
    def pos(self):
        return self._gripper_pos_reading

    @property
    def last_binary(self):
        return self._last_gripper_bin
