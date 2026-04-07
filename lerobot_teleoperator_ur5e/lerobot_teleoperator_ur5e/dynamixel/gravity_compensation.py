"""
Gravity compensation background thread for Dynamixel servos.

Runs as a background thread. Uses driver's cached position and current data
(zero bus cost). Only writes to bus for PID changes and goal position updates.

Allows hand-dragging of robot arms with compliant hold behavior.
"""

import time
import threading
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# XC430 Control Table addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_PWM = 100
ADDR_PROFILE_ACC = 108
ADDR_PROFILE_VEL = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
POSITION_CONTROL_MODE = 3
EXTENDED_POSITION_CONTROL_MODE = 4

# Drag / hold parameters
SETTLE_DEG = 0.5
SETTLE_FRAMES = 3
CURRENT_LIMIT_MA = 700

DEFAULT_P = 140
DEFAULT_I = 0
DEFAULT_D = 30
DRAG_P = 80
DRAG_D = 10
DEFAULT_PWM = 885
DEFAULT_PWM_HEAVY = 700
HEAVY_P = 350
DRAG_CURRENT_MA = 80


DEG_TO_UNITS_SCALE = 4096.0 / 360.0
UNITS_TO_DEG_SCALE = 360.0 / 4096.0


def deg_to_units(deg: float) -> int:
    return int(deg * DEG_TO_UNITS_SCALE)


def units_to_deg(units: int) -> float:
    return units * UNITS_TO_DEG_SCALE


DEFAULT_DRAG_DEG = 3.0


DEFAULT_WRITE_TH_DEG = 0.18  # ~2 encoder units; skip goal write if position moved less than this


def _load_joint_pid(arm_key: str, jid: int, joint_ids: list, cfg: dict) -> tuple:
    """Return (p, d, pwm, drag_ma, drag_deg) from cfg for the given arm, else hardcoded defaults."""
    saved = cfg[arm_key].get("joint_pid")
    if saved:
        if isinstance(saved.get("p"), list):
            idx = joint_ids.index(jid)
            dm = saved["drag_ma"][idx] if isinstance(saved.get("drag_ma"), list) else DRAG_CURRENT_MA
            dd = saved["drag_deg"][idx] if isinstance(saved.get("drag_deg"), list) else DEFAULT_DRAG_DEG
            return saved["p"][idx], saved["d"][idx], saved["pwm"][idx], dm, dd
        if jid in saved:
            e = saved[jid]
            return e["p"], e["d"], e["pwm"], e.get("drag_ma", DRAG_CURRENT_MA), e.get("drag_deg", DEFAULT_DRAG_DEG)
    pwm = DEFAULT_PWM_HEAVY if jid in joint_ids[:2] else DEFAULT_PWM
    p = HEAVY_P if jid in joint_ids[:2] else DEFAULT_P
    return p, DEFAULT_D, pwm, DRAG_CURRENT_MA, DEFAULT_DRAG_DEG


class GravityCompensationThread:
    """
    Background thread for gravity compensation.
    Uses GroupSyncRead for synchronized position and current readings.
    Only writes to bus for PID/profile changes and goal position updates.
    """

    def __init__(self, driver, arm_key: str, joint_ids: list,
                 servo_cfg_path: str = None, loop_hz: int = 50):
        self._driver = driver
        self._ph = driver._portHandler
        self._pkt = driver._packetHandler
        self._arm_key = arm_key
        self._joint_ids = joint_ids
        self._period = 1.0 / loop_hz
        self._loop_hz = loop_hz
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()
        self._thread = None

        # Load servo config
        if servo_cfg_path is None:
            servo_cfg_path = Path(__file__).parent / "servo_cfg.yaml"

        with open(servo_cfg_path, "r") as f:
            self._cfg = yaml.safe_load(f)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"[GCOMP] Background gravity compensation started for {self._arm_key}")

    def pause(self):
        """Pause gravity compensation (e.g., during teleoperation)"""
        self._paused.set()
        logger.info(f"[GCOMP] Paused for {self._arm_key}")

    def resume(self):
        """Resume gravity compensation"""
        self._paused.clear()
        logger.info(f"[GCOMP] Resumed for {self._arm_key}")

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logger.warning(f"[GCOMP] Thread for {self._arm_key} did not exit in time")
            # Thread didn't exit cleanly — force torque disable as safety net
            for jid in self._joint_ids:
                jid_ = jid
                self._driver._write_queue.put(
                    lambda j=jid_: self._pkt.write1ByteTxRx(self._ph, j, ADDR_TORQUE_ENABLE, 0)
                )

        self._thread = None
        logger.info(f"[GCOMP] Gravity compensation stopped for {self._arm_key}")

    def _enqueue(self, fn):
        """Queue a write to be executed by the driver's bus thread."""
        self._driver._write_queue.put(fn)

    def _w1(self, jid, addr, val):
        self._enqueue(lambda j=jid, a=addr, v=val: self._pkt.write1ByteTxRx(self._ph, j, a, v))

    def _w2(self, jid, addr, val):
        self._enqueue(lambda j=jid, a=addr, v=val: self._pkt.write2ByteTxRx(self._ph, j, a, v))

    def _w4(self, jid, addr, val):
        self._enqueue(lambda j=jid, a=addr, v=val: self._pkt.write4ByteTxRx(self._ph, j, a, v))

    def _flush_goals_batch(self, pending_goals: dict):
        """Batch all pending goal position writes into one GroupSyncWrite packet.

        Replaces N individual write4ByteTxRx calls (N×~4ms at 57600 baud) with a
        single GroupSyncWrite (~4ms regardless of N joints). This prevents the write
        queue from starving the position read, keeping _joint_angles cache fresh.

        Only use for continuous tracking writes — transition writes (HOLD↔DRAG) must
        remain individual _w4 calls to preserve goal-before-PID ordering.
        """
        if not pending_goals:
            return
        updates = list(pending_goals.items())
        def _write():
            for jid, pos_units in updates:
                u = pos_units & 0xFFFFFFFF
                param = [u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, (u >> 24) & 0xFF]
                self._driver._groupSyncWrite.addParam(jid, param)
            self._driver._groupSyncWrite.txPacket()
            self._driver._groupSyncWrite.clearParam()
        self._driver._write_queue.put(_write)

    def _run(self):
        try:
            # Load PID parameters for each joint
            pids = {}
            drag_ma = {}
            drag_deg = {}
            for jid in self._joint_ids:
                p, d, pwm, dm, dd = _load_joint_pid(self._arm_key, jid, self._joint_ids, self._cfg)
                pids[jid] = [p, DEFAULT_I, d, pwm]
                drag_ma[jid] = dm
                drag_deg[jid] = dd

            # Drag mode and write threshold from config
            arm_cfg = self._cfg[self._arm_key]
            joint_pid_cfg = arm_cfg.get("joint_pid", {})
            drag_mode = joint_pid_cfg.get("drag_mode", "current")
            write_th_deg = joint_pid_cfg.get("write_th_deg", DEFAULT_WRITE_TH_DEG)

            # Pre-compute index list (ordered same as _joint_ids) for direct access
            driver_id_list = list(self._driver._ids)
            servo_indices = []
            for jid in self._joint_ids:
                try:
                    servo_indices.append(driver_id_list.index(jid))
                except ValueError:
                    logger.error(f"[GCOMP] Joint ID {jid} not found in driver IDs {self._driver._ids}")
                    return

            # Initialize: set extended position mode and apply PID
            for jid in self._joint_ids:
                self._w1(jid, ADDR_TORQUE_ENABLE, 0)
                self._w1(jid, ADDR_OPERATING_MODE, EXTENDED_POSITION_CONTROL_MODE)
                p, i, d, pwm = pids[jid]
                self._w2(jid, ADDR_POSITION_P_GAIN, p)
                self._w2(jid, ADDR_POSITION_I_GAIN, i)
                self._w2(jid, ADDR_POSITION_D_GAIN, d)
                self._w2(jid, ADDR_GOAL_PWM, pwm)
                self._w4(jid, ADDR_PROFILE_VEL, 100)
                self._w4(jid, ADDR_PROFILE_ACC, 50)

            # Read initial positions from driver's cached data
            pos = {}
            goal = {}
            # Wait for driver to have valid data and init writes to flush
            while self._driver._joint_angles is None or self._driver._currents is None:
                time.sleep(0.01)
            # Wait a bit for queued init writes to be drained
            time.sleep(0.1)

            for i, jid in enumerate(self._joint_ids):
                idx = servo_indices[i]
                raw_pos = int(self._driver._joint_angles[idx])
                p_deg = units_to_deg(raw_pos)
                pos[jid] = p_deg
                goal[jid] = p_deg
                self._w4(jid, ADDR_GOAL_POSITION, deg_to_units(p_deg))

            # Enable torque
            for jid in self._joint_ids:
                self._w1(jid, ADDR_TORQUE_ENABLE, 1)

            # Wait for torque enable to flush
            time.sleep(0.05)
            logger.info(f"[GCOMP] Initialized for {self._arm_key}, torque enabled")

            # State tracking
            cur = {jid: 0 for jid in self._joint_ids}
            state = {jid: "HOLD" for jid in self._joint_ids}
            settle_c = {jid: 0 for jid in self._joint_ids}
            hold_cooldown = {jid: 0 for jid in self._joint_ids}
            HOLD_COOLDOWN_FRAMES = max(int(0.5 * self._loop_hz), 3)
            last_pos = dict(pos)

            loop_next = time.perf_counter()

            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.1)
                    continue

                # Collect goal position updates this tick — flushed as one GroupSyncWrite
                # at the end of the tick to minimise bus write overhead.
                # Transition writes (HOLD↔DRAG) bypass this dict and use _w4 directly
                # to preserve goal-before-PID ordering.
                pending_goals = {}

                # Read positions and currents from driver's cached data (zero bus cost)
                # Driver already sign-converts, so no signed_16/signed_32 needed.
                joint_angles = self._driver._joint_angles
                currents = self._driver._currents
                for i, jid in enumerate(self._joint_ids):
                    idx = servo_indices[i]
                    pos[jid] = units_to_deg(int(joint_angles[idx]))
                    cur[jid] = int(currents[idx])

                    # Current protection: snap goal to current pos, write only if moved enough
                    if abs(cur[jid]) > CURRENT_LIMIT_MA:
                        if abs(pos[jid] - goal[jid]) > write_th_deg:
                            goal[jid] = pos[jid]
                            pending_goals[jid] = deg_to_units(pos[jid])
                        if state[jid] == "HOLD":
                            logger.warning(f"[GCOMP] J{jid} current limit ({cur[jid]}mA), snapping goal to current pos")

                # Snapshot state before running state machine
                prev_state = dict(state)

                # State machine
                for jid in self._joint_ids:
                    delta = abs(pos[jid] - last_pos[jid])

                    if state[jid] == "HOLD":
                        if hold_cooldown[jid] > 0:
                            hold_cooldown[jid] -= 1
                        else:
                            # Check drag entry conditions based on drag_mode
                            current_trigger = abs(cur[jid]) > drag_ma[jid]
                            position_trigger = abs(pos[jid] - goal[jid]) > drag_deg[jid]

                            if drag_mode == "current":
                                enter_drag = current_trigger
                            elif drag_mode == "position":
                                enter_drag = position_trigger
                            elif drag_mode == "either":
                                enter_drag = current_trigger or position_trigger
                            else:  # "both" or unknown
                                enter_drag = current_trigger and position_trigger

                            if enter_drag:
                                state[jid] = "DRAG"
                                settle_c[jid] = 0
                                goal[jid] = pos[jid]

                    elif state[jid] == "DRAG":
                        # Continuous goal tracking — write only if moved beyond threshold
                        if abs(pos[jid] - goal[jid]) > write_th_deg:
                            logger.debug(f"[GCOMP] J{jid} DRAG write: pos={pos[jid]:.2f} goal_was={goal[jid]:.2f}")
                            goal[jid] = pos[jid]
                            pending_goals[jid] = deg_to_units(pos[jid])

                        if delta < SETTLE_DEG:
                            settle_c[jid] += 1
                            if settle_c[jid] >= SETTLE_FRAMES:
                                state[jid] = "HOLD"
                                hold_cooldown[jid] = HOLD_COOLDOWN_FRAMES
                                goal[jid] = pos[jid]
                                pending_goals.pop(jid, None)
                        else:
                            settle_c[jid] = 0

                    last_pos[jid] = pos[jid]

                # Enqueue goal + PID/profile writes only for joints that changed state.
                # Goal write MUST be queued before PID writes to preserve ordering.
                for jid in self._joint_ids:
                    if state[jid] == prev_state[jid]:
                        continue
                    goal_units = deg_to_units(goal[jid])
                    pending_goals.pop(jid, None)  # remove from batch; write individually
                    self._w4(jid, ADDR_GOAL_POSITION, goal_units)
                    if state[jid] == "DRAG":
                        self._w4(jid, ADDR_PROFILE_VEL, 0)
                        self._w4(jid, ADDR_PROFILE_ACC, 0)
                        self._w2(jid, ADDR_POSITION_P_GAIN, DRAG_P)
                        self._w2(jid, ADDR_POSITION_D_GAIN, DRAG_D)
                    else:  # HOLD
                        self._w4(jid, ADDR_PROFILE_VEL, 100)
                        self._w4(jid, ADDR_PROFILE_ACC, 50)
                        self._w2(jid, ADDR_POSITION_P_GAIN, pids[jid][0])
                        self._w2(jid, ADDR_POSITION_D_GAIN, pids[jid][2])

                # Flush all routine goal position updates as one GroupSyncWrite packet.
                # This runs AFTER transition writes are already queued, so ordering is safe.
                self._flush_goals_batch(pending_goals)

                loop_next += self._period
                time.sleep(max(0, loop_next - time.perf_counter()))

        except Exception as e:
            logger.error(f"[GCOMP] Error in gravity compensation thread: {e}")
        finally:
            for jid in self._joint_ids:
                jid_ = jid
                self._driver._write_queue.put(
                    lambda j=jid_: self._pkt.write1ByteTxRx(self._ph, j, ADDR_TORQUE_ENABLE, 0)
                )
            logger.info(f"[GCOMP] Cleanup complete for {self._arm_key}")
