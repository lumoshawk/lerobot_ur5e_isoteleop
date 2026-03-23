"""
Virtual trigger emulator for Dynamixel XC430 servo.

Runs as a background thread. Reads position from the driver's cached
GroupSyncRead data (zero bus cost), only writes to bus for state changes
and goal position updates.
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# XC430 Control Table addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_PWM = 100
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
POSITION_CONTROL_MODE = 3


def _deg_to_units(deg: float) -> int:
    return int((deg / 360.0) * 4096.0)

def _units_to_deg(units: int) -> float:
    return (units * 360.0) / 4096.0

def _signed_16(v: int) -> int:
    return v - 65536 if v > 32767 else v

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class TriggerParams:
    wall_deg: float = 38
    break_deg: float = 39
    overtravel_deg: float = 40
    p_return: int = 900
    d_return: int = 25
    prof_vel_return: int = 700
    prof_acc_return: int = 600
    pwm_return: int = 885
    return_done_deg: float = 1.5
    return_done_ma: int = 30
    release_arm_deg: float = 4.0
    release_vel_threshold: float = 12.0
    release_confirm_frames: int = 2
    release_drop_deg: float = 1.0
    passive_return_arm_deg: float = 3.0
    passive_return_dq_th: float = -10
    prof_vel_passive: int = 500
    prof_acc_passive: int = 400
    p_passive: int = 700
    d_passive: int = 20
    p_takeup: int = 50
    d_takeup: int = 8
    p_wall: int = 700
    d_wall: int = 45
    p_overtravel: int = 250
    d_overtravel: int = 10
    i_gain: int = 0
    prof_vel_takeup: int = 100
    prof_acc_takeup: int = 50
    prof_vel_wall: int = 35
    prof_acc_wall: int = 20
    prof_vel_break: int = 180
    prof_acc_break: int = 120
    break_jump_deg: float = 4.0
    break_hold_ms: int = 50
    pwm_takeup: int = 885
    pwm_wall: int = 650
    pwm_break: int = 700
    current_limit_ma: int = 450


class TriggerEmulator:
    """State machine: TAKEUP -> WALL -> BREAK -> OVERTRAVEL -> ACTIVE_RETURN -> TAKEUP"""

    def __init__(self, params: TriggerParams, rest_deg_abs: float):
        self.p = params
        self.rest_deg_abs = rest_deg_abs
        self.state = "TAKEUP"
        self.broken = False
        self.break_until = 0.0
        self.peak_q_rel = 0.0
        self.release_counter = 0
        self.returning = False
        self.last_q_abs = None
        self.last_t = None

    def update(self, q_abs_deg: float, current_ma: int = 0):
        now = time.perf_counter()
        if self.last_q_abs is None:
            self.last_q_abs = q_abs_deg
            self.last_t = now
        dt = max(1e-3, now - self.last_t)
        dq = (q_abs_deg - self.last_q_abs) / dt
        self.last_q_abs = q_abs_deg
        self.last_t = now
        q_rel = q_abs_deg - self.rest_deg_abs

        if self.broken:
            self.peak_q_rel = max(self.peak_q_rel, q_rel)
        else:
            self.peak_q_rel = q_rel

        releasing_raw = (
            q_rel > self.p.release_arm_deg
            and (self.peak_q_rel - q_rel) > self.p.release_drop_deg
            and dq < -self.p.release_vel_threshold
        )
        self.release_counter = self.release_counter + 1 if releasing_raw else 0
        releasing = self.release_counter >= self.p.release_confirm_frames

        if not self.broken:
            if q_rel < self.p.wall_deg:
                self.state = "TAKEUP"
            elif q_rel < self.p.break_deg:
                self.state = "WALL"
            else:
                self.state = "BREAK"
                self.broken = True
                self.break_until = now + self.p.break_hold_ms / 1000.0
                self.peak_q_rel = q_rel
        else:
            if self.returning or releasing:
                self.returning = True
                self.state = "ACTIVE_RETURN"
            elif now < self.break_until:
                self.state = "BREAK"
            else:
                self.state = "OVERTRAVEL"

        p = self.p
        if self.state == "TAKEUP":
            if q_rel > p.passive_return_arm_deg and dq < p.passive_return_dq_th:
                return {"goal_deg": self.rest_deg_abs, "p_gain": p.p_passive,
                        "i_gain": p.i_gain, "d_gain": p.d_passive,
                        "profile_vel": p.prof_vel_passive, "profile_acc": p.prof_acc_passive,
                        "goal_pwm": p.pwm_takeup}
            return {"goal_deg": self.rest_deg_abs, "p_gain": p.p_takeup,
                    "i_gain": p.i_gain, "d_gain": p.d_takeup,
                    "profile_vel": p.prof_vel_takeup, "profile_acc": p.prof_acc_takeup,
                    "goal_pwm": p.pwm_takeup}
        if self.state == "WALL":
            return {"goal_deg": self.rest_deg_abs + p.wall_deg, "p_gain": p.p_wall,
                    "i_gain": p.i_gain, "d_gain": p.d_wall,
                    "profile_vel": p.prof_vel_wall, "profile_acc": p.prof_acc_wall,
                    "goal_pwm": p.pwm_wall}
        if self.state == "BREAK":
            return {"goal_deg": self.rest_deg_abs + min(p.overtravel_deg, p.break_deg + p.break_jump_deg),
                    "p_gain": int(p.p_wall * 0.35), "i_gain": p.i_gain, "d_gain": p.d_wall,
                    "profile_vel": p.prof_vel_break, "profile_acc": p.prof_acc_break,
                    "goal_pwm": p.pwm_break}
        if self.state == "OVERTRAVEL":
            return {"goal_deg": self.rest_deg_abs + p.break_deg, "p_gain": p.p_overtravel,
                    "i_gain": p.i_gain, "d_gain": p.d_overtravel,
                    "profile_vel": p.prof_vel_takeup, "profile_acc": p.prof_acc_takeup,
                    "goal_pwm": p.pwm_break}
        # ACTIVE_RETURN
        if abs(q_rel) < p.return_done_deg and abs(current_ma) < p.return_done_ma:
            self.broken = False
            self.returning = False
            self.release_counter = 0
            self.peak_q_rel = q_rel
            self.state = "TAKEUP"
        return {"goal_deg": self.rest_deg_abs, "p_gain": p.p_return,
                "i_gain": p.i_gain, "d_gain": p.d_return,
                "profile_vel": p.prof_vel_return, "profile_acc": p.prof_acc_return,
                "goal_pwm": p.pwm_return}


class TriggerThread:
    """
    Reads servo position from the driver's cached GroupSyncRead data (no bus reads).
    Only writes to bus for PID/profile changes and goal position updates.
    """

    def __init__(self, driver, servo_id: int = 13,
                 rest_deg: float = 220.0, sign: int = 1, loop_hz: int = 10):
        self._driver = driver
        self._ph = driver._portHandler
        self._pkt = driver._packetHandler
        self._lock = driver._lock
        self._sid = servo_id
        self._rest_deg = rest_deg
        self._sign = sign
        self._period = 1.0 / loop_hz
        self._stop = threading.Event()
        self._thread = None

        # Find index of this servo in the driver's ID list
        self._servo_idx = list(driver._ids).index(servo_id)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"[TRIGGER] Background trigger started on servo {self._sid}")

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        # Disable torque on exit
        with self._lock:
            self._pkt.write1ByteTxRx(
                self._ph, self._sid, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self._thread = None

    def _w1(self, addr, val):
        try:
            with self._lock:
                self._pkt.write1ByteTxRx(self._ph, self._sid, addr, val)
        except Exception:
            pass

    def _w2(self, addr, val):
        try:
            with self._lock:
                self._pkt.write2ByteTxRx(self._ph, self._sid, addr, val)
        except Exception:
            pass

    def _w4(self, addr, val):
        try:
            with self._lock:
                self._pkt.write4ByteTxRx(self._ph, self._sid, addr, val)
        except Exception:
            pass

    def _get_pos_deg(self):
        """Read position from driver's cached GroupSyncRead data. No bus access."""
        raw = self._driver._joint_angles
        if raw is None:
            return None
        # raw is in dynamixel units (int), convert to degrees
        return float(raw[self._servo_idx]) / 2048.0 * 180.0

    def _run(self):
        # Init servo: position mode + torque on
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self._w1(ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
        self._w1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        # Home to rest position
        self._w4(ADDR_PROFILE_VELOCITY, 80)
        self._w4(ADDR_PROFILE_ACCELERATION, 40)
        self._w4(ADDR_GOAL_POSITION, _deg_to_units(self._rest_deg))

        # Wait for homing using cached data
        deadline = time.time() + 5.0
        while time.time() < deadline and not self._stop.is_set():
            pos_deg = self._get_pos_deg()
            if pos_deg is not None and abs(pos_deg - self._rest_deg) < 1.0:
                break
            time.sleep(0.02)

        emu = TriggerEmulator(TriggerParams(), rest_deg_abs=self._rest_deg)
        last_cfg = None
        last_goal_units = None
        loop_next = time.perf_counter()

        while not self._stop.is_set():
            # Read from cached data — zero bus cost
            pos_deg = self._get_pos_deg()
            if pos_deg is None:
                loop_next += self._period
                time.sleep(max(0, loop_next - time.perf_counter()))
                continue

            emu_pos = self._rest_deg + self._sign * (pos_deg - self._rest_deg)
            cmd = emu.update(emu_pos, current_ma=0)

            # Only write PID/profile when config changes
            cfg_tuple = (cmd["p_gain"], cmd["i_gain"], cmd["d_gain"],
                         cmd["profile_vel"], cmd["profile_acc"], cmd["goal_pwm"])
            if cfg_tuple != last_cfg:
                self._w2(ADDR_POSITION_P_GAIN, cmd["p_gain"])
                self._w2(ADDR_POSITION_I_GAIN, cmd["i_gain"])
                self._w2(ADDR_POSITION_D_GAIN, cmd["d_gain"])
                self._w4(ADDR_PROFILE_VELOCITY, cmd["profile_vel"])
                self._w4(ADDR_PROFILE_ACCELERATION, cmd["profile_acc"])
                self._w2(ADDR_GOAL_PWM, cmd["goal_pwm"])
                last_cfg = cfg_tuple

            # Only write goal position when it changes
            phys_goal = self._rest_deg + self._sign * (cmd["goal_deg"] - self._rest_deg)
            goal_units = _deg_to_units(_clamp(phys_goal, 0.0, 360.0))
            if goal_units != last_goal_units:
                self._w4(ADDR_GOAL_POSITION, goal_units)
                last_goal_units = goal_units

            loop_next += self._period
            time.sleep(max(0, loop_next - time.perf_counter()))

