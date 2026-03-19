# Add Freedrive (Dragging Demo) Mode Toggle During Teleoperation

## Context
During teleoperation, the user wants to temporarily switch the UR5e into freedrive/teach mode (physically drag the arm to demonstrate) via keyboard shortcuts, then return to start position and resume teleop. Partial episode data is discarded when freedrive is triggered mid-episode.

- **'m' key**: Stop teleop, enter freedrive (dragging demo) mode
- **'b' key**: Exit freedrive, move UR5e back to start position, resume teleop
- Available anytime: during recording AND between episodes

## File to modify
- `scripts/core/run_record.py`

## Key APIs (ur_rtde RTDEControlInterface)
- `rtde_c.teachMode()` — enable freedrive
- `rtde_c.endTeachMode()` — disable freedrive
- `rtde_c.moveJ(target_rad, speed, accel)` — move to start position
- Single arm access: `robot._arm["rtde_c"]`, start pos: `robot.config.start_position`
- Dual arm access: `robot.left_arm._arm["rtde_c"]` / `robot.right_arm._arm["rtde_c"]`

## Implementation

### 1. Add imports
Add `import time` and `import numpy as np` at top of `run_record.py`.

### 2. Add helper functions (before `run_record()`)

```python
def enter_freedrive(robot, is_dual_arm):
    if is_dual_arm:
        robot.left_arm._arm["rtde_c"].teachMode()
        robot.right_arm._arm["rtde_c"].teachMode()
    else:
        robot._arm["rtde_c"].teachMode()

def exit_freedrive_and_return(robot, is_dual_arm):
    if is_dual_arm:
        robot.left_arm._arm["rtde_c"].endTeachMode()
        robot.right_arm._arm["rtde_c"].endTeachMode()
        robot.left_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.left_arm.config.start_position).tolist(), 0.5, 0.5)
        robot.right_arm._arm["rtde_c"].moveJ(np.deg2rad(robot.right_arm.config.start_position).tolist(), 0.5, 0.5)
    else:
        robot._arm["rtde_c"].endTeachMode()
        robot._arm["rtde_c"].moveJ(np.deg2rad(robot.config.start_position).tolist(), 0.5, 0.5)
```

### 3. Add freedrive state dict and key handlers
In `run_record()`, after the `p_listener` setup (~line 403), add:

```python
freedrive_state = {"active": False, "request_enter": False, "request_exit": False}
```

Extend the existing `_on_p_press` handler (or add a new handler in the same pynput Listener) to also handle 'm' and 'b':
- 'm' → `freedrive_state["request_enter"] = True; events["exit_early"] = True`
- 'b' → `freedrive_state["request_exit"] = True`

### 4. Add freedrive handling helper

```python
def handle_freedrive_if_requested(freedrive_state, events, robot, is_dual_arm, dataset):
    """Check and handle freedrive request. Returns True if freedrive was activated."""
    if not freedrive_state["request_enter"]:
        return False
    freedrive_state["request_enter"] = False
    events["exit_early"] = False

    logging.info("====== [FREEDRIVE] Entering dragging demo mode. Press 'b' to return. ======")
    enter_freedrive(robot, is_dual_arm)
    freedrive_state["active"] = True

    while not freedrive_state["request_exit"] and not events["stop_recording"]:
        time.sleep(0.1)

    freedrive_state["request_exit"] = False
    logging.info("====== [FREEDRIVE] Returning to start position... ======")
    exit_freedrive_and_return(robot, is_dual_arm)
    freedrive_state["active"] = False
    logging.info("====== [FREEDRIVE] Resumed. Ready for teleop. ======")

    if dataset is not None:
        dataset.clear_episode_buffer()
    return True
```

### 5. Integrate into episode loop
In the `while episode_idx < ...` loop (~line 407), after `record_loop(...)` returns:

```python
record_loop(...)

# If freedrive was requested mid-episode, handle it and restart episode
if handle_freedrive_if_requested(freedrive_state, events, robot, record_cfg.is_dual_arm, dataset):
    continue  # restart this episode from scratch
```

Also add freedrive check in the "wait for Enter to reset" block (~line 434). Replace the simple `input()` wait with a loop that also checks `freedrive_state["request_enter"]`:

```python
# Between episodes: check for freedrive during wait
while True:
    if freedrive_state["request_enter"]:
        handle_freedrive_if_requested(freedrive_state, events, robot, record_cfg.is_dual_arm, None)
        continue
    termios.tcflush(sys.stdin, termios.TCIFLUSH)
    user_input = input("====== [WAIT] Press Enter to reset ======")
    if user_input == "":
        break
```

Note: Since `input()` is blocking, the freedrive check between episodes will only trigger if 'm' was pressed before `input()` blocks. This is acceptable — during active recording, 'm' will break out of `record_loop` via `exit_early` and be handled immediately.

### 6. Cleanup
Stop the pynput listener in the existing cleanup section (already handled by `p_listener.stop()`). The freedrive key handling is part of the same listener, so no extra cleanup needed.

## Verification
1. Run the recording script
2. During teleop recording, press 'm' — episode should stop, arm enters freedrive (draggable)
3. Drag the arm around
4. Press 'b' — arm moves back to start position, partial episode discarded, teleop resumes with fresh episode
5. Between episodes, press 'm' during reset phase — same freedrive behavior
6. Verify 'p' key still works for position snapshots
7. Test with both single-arm and dual-arm configs
