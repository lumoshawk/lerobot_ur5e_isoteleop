"""
Set Dynamixel servo baudrate.

Reads current baudrate and target baudrate from cfg.yaml.
Connects at the current baudrate, reprograms all servos, then verifies
by reconnecting at the new baudrate.

Usage:
    python scripts/utils/set_dynamixel_baudrate.py --from-baud 57600 --to-baud 1000000
    python scripts/utils/set_dynamixel_baudrate.py  # uses cfg.yaml baudrate as target, 57600 as source
"""

import argparse
import time
import yaml
from pathlib import Path
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# Dynamixel X/MX series register
ADDR_BAUDRATE = 8
PROTOCOL_VERSION = 2.0

# Register value → baud rate mapping (from datasheet)
BAUD_VALUE_MAP = {
    9600:    0,
    57600:   1,
    115200:  2,
    1000000: 3,
    2000000: 4,
    3000000: 5,
    4000000: 6,
}
VALUE_BAUD_MAP = {v: k for k, v in BAUD_VALUE_MAP.items()}


def load_cfg():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "cfg.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_all_joint_ids(cfg):
    record = cfg["record"]
    teleop = record["teleop"]
    if record.get("dual_arm_mode", False):
        ids = list(teleop["left_arm"]["joint_ids"]) + list(teleop["right_arm"]["joint_ids"])
        # include gripper IDs if present
        for arm in ("left_arm", "right_arm"):
            gc = teleop[arm].get("gripper_config")
            if gc:
                ids.append(int(gc[0]))
        return teleop["port"], ids
    else:
        dxl = teleop["dynamixel_config"]
        ids = list(dxl["joint_ids"])
        gc = dxl.get("gripper_config")
        if gc:
            ids.append(int(gc[0]))
        return dxl["port"], ids


def reprogram(port_name, joint_ids, from_baud, to_baud):
    if to_baud not in BAUD_VALUE_MAP:
        raise ValueError(f"Unsupported target baud rate {to_baud}. Supported: {sorted(BAUD_VALUE_MAP)}")
    if from_baud not in BAUD_VALUE_MAP:
        raise ValueError(f"Unsupported source baud rate {from_baud}. Supported: {sorted(BAUD_VALUE_MAP)}")

    target_val = BAUD_VALUE_MAP[to_baud]

    port = PortHandler(port_name)
    ph = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError(f"Failed to open port {port_name}")
    if not port.setBaudRate(from_baud):
        raise RuntimeError(f"Failed to set baud rate to {from_baud}")

    print(f"Connected at {from_baud} bps. Reprogramming {len(joint_ids)} servos to {to_baud} bps...")

    failed = []
    for jid in joint_ids:
        # Disable torque first (required to write EEPROM on most X-series)
        ph.write1ByteTxRx(port, jid, 64, 0)  # ADDR_TORQUE_ENABLE = 64
        time.sleep(0.01)

        result, error = ph.write1ByteTxRx(port, jid, ADDR_BAUDRATE, target_val)
        if result != COMM_SUCCESS:
            print(f"  ID {jid:3d}: FAILED (comm result={result})")
            failed.append(jid)
        elif error != 0:
            print(f"  ID {jid:3d}: FAILED (servo error={error})")
            failed.append(jid)
        else:
            print(f"  ID {jid:3d}: OK")
        time.sleep(0.02)

    port.closePort()

    if failed:
        print(f"\nFailed IDs: {failed}")
        return False

    # Verify by reconnecting at new baud
    print(f"\nVerifying — reconnecting at {to_baud} bps...")
    time.sleep(0.3)

    port2 = PortHandler(port_name)
    ph2 = PacketHandler(PROTOCOL_VERSION)
    port2.openPort()
    port2.setBaudRate(to_baud)

    ok = []
    fail2 = []
    for jid in joint_ids:
        model, result, error = ph2.ping(port2, jid)
        if result == COMM_SUCCESS:
            print(f"  ID {jid:3d}: ping OK (model={model})")
            ok.append(jid)
        else:
            print(f"  ID {jid:3d}: ping FAILED (result={result})")
            fail2.append(jid)

    port2.closePort()

    if fail2:
        print(f"\nVerification failed for IDs: {fail2}")
        return False

    print(f"\nAll {len(ok)} servos successfully reprogrammed to {to_baud} bps.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reprogram Dynamixel servo baudrate")
    parser.add_argument("--from-baud", type=int, default=None,
                        help="Current servo baudrate (default: 57600)")
    parser.add_argument("--to-baud", type=int, default=None,
                        help="Target baudrate (default: read from cfg.yaml teleop.baudrate)")
    args = parser.parse_args()

    cfg = load_cfg()
    port_name, joint_ids = get_all_joint_ids(cfg)

    from_baud = args.from_baud if args.from_baud is not None else 1000000
    to_baud = args.to_baud if args.to_baud is not None else cfg["record"]["teleop"].get("baudrate", 57600)

    print(f"Port:        {port_name}")
    print(f"Servo IDs:   {joint_ids}")
    print(f"From baud:   {from_baud}")
    print(f"To baud:     {to_baud}")

    if from_baud == to_baud:
        print("Source and target baudrate are the same — nothing to do.")
        return

    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    success = reprogram(port_name, joint_ids, from_baud, to_baud)
    if not success:
        print("\nSome servos failed. Check connections and retry.")


if __name__ == "__main__":
    main()
