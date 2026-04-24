#!/usr/bin/env python3

import zmq
import threading
import time
import json
import subprocess
from pathlib import Path

def get_git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )

root_path = get_git_root()

# --- Shared State ---
camera_state = {
    "fps": 1000.0,
    "gain": 1,
    "status": "operational",
    "mode": "globalresetburst",
    "resetwidth": 20,
    "nbreadworeset": 200,
    "cropping": False,
    "cropping rows": "0-255",
    "cropping columns": "0-319"
}


phasemask_positions = {}
for beam in [1,2,3,4]:
    with open(root_path / f"baldrapp/apps/paranal_simulator/fake_configs/phase_positions_beam{beam}_2025-01-01T00-00-00.json") as f:
        phasemask_positions[beam] = json.load( f ) 

#/home/rtc/Documents/dcs/simulation/fake_configs/phase_positions_beam4_2025-07-11T15-04-05.json

with open(root_path / "baldrapp/apps/paranal_simulator/fake_configs/mds_state.json") as f:
    mds_state = json.load( f )

motor_state = {d["name"]: d for d in mds_state}


# --- Camera Command Handler ---
def handle_camera_command(cmd: str) -> str:
    #print(f"recieved: {cmd}")
    cmd = cmd.strip().lower()

    if cmd.startswith("cli "):
        cmd = cmd[4:].strip().strip('"')

    if cmd in ("fps", "fps raw"):
        return f"{camera_state['fps']:.2f}"
    elif cmd in ("gain", "gain raw"):
        return f"{camera_state['gain']:.2f}"
    elif cmd in ("status", "status raw"):
        return camera_state["status"]
    elif cmd in ("mode", "mode raw"):
        return camera_state["mode"]
    elif cmd in ("resetwidth", "resetwidth raw"):
        return str(camera_state["resetwidth"])
    elif cmd in ("nbreadworeset", "nbreadworeset raw"):
        return str(camera_state["nbreadworeset"])
    elif cmd in ("cropping", "cropping raw"):
        return "active" if camera_state["cropping"] else "inactive"
    elif cmd in ("cropping rows", "cropping rows raw"):
        return camera_state["cropping rows"]
    elif cmd in ("cropping columns", "cropping columns raw"):
        return camera_state["cropping columns"]

    elif cmd.startswith("set fps "):
        try:
            val = float(cmd.split()[-1])
            camera_state["fps"] = val
            return f"fps set to {val}"
        except Exception:
            return "Invalid fps value"
    elif cmd.startswith("set gain "):
        try:
            val = float(cmd.split()[-1])
            camera_state["gain"] = val
            return f"gain set to {val}"
        except Exception:
            return "Invalid gain value"
    elif cmd.startswith("set mode "):
        camera_state["mode"] = cmd[len("set mode "):]
        return f"mode set to {camera_state['mode']}"
    elif cmd.startswith("set resetwidth "):
        try:
            val = int(cmd.split()[-1])
            camera_state["resetwidth"] = val
            return f"resetwidth set to {val}"
        except Exception:
            return "Invalid resetwidth value"
    elif cmd.startswith("set nbreadworeset "):
        try:
            val = int(cmd.split()[-1])
            camera_state["nbreadworeset"] = val
            return f"nbreadworeset set to {val}"
        except Exception:
            return "Invalid nbreadworeset value"
    elif cmd == "set cropping on":
        camera_state["cropping"] = True
        return "cropping enabled"
    elif cmd == "set cropping off":
        camera_state["cropping"] = False
        return "cropping disabled"
    elif cmd.startswith("set cropping rows "):
        camera_state["cropping rows"] = cmd[len("set cropping rows "):]
        return f"cropping rows set to {camera_state['cropping rows']}"
    elif cmd.startswith("set cropping columns "):
        camera_state["cropping columns"] = cmd[len("set cropping columns "):]
        return f"cropping columns set to {camera_state['cropping columns']}"

    return f"Unknown or unsupported command: {cmd}"

# --- MDS Command Handler ---

def handle_mds_command(cmd: str) -> str:
    cmd = cmd.strip()
    tokens = cmd.split()
    
    if cmd.lower() == "on sbb":
        mds_state["SBB"] = True
        return "OK"
    elif cmd.lower() == "off sbb":
        mds_state["SBB"] = False
        return "OK"

    elif tokens[0].lower() == "read":
        if len(tokens) != 2:
            return "NACK: Usage is 'read <axis>'"
        axis = tokens[1]
        if axis in motor_state and motor_state[axis]["is_connected"]:
            return f"{motor_state[axis]['position']:.6f}"
        else:
            return f"NACK: Axis '{axis}' not found or not connected"

    elif tokens[0].lower() == "moverel":
        if len(tokens) != 3:
            return "NACK: Usage is 'moverel <axis> <delta>'"
        axis, delta = tokens[1], tokens[2]
        try:
            delta = float(delta)
            if axis in motor_state and motor_state[axis]["is_connected"]:
                motor_state[axis]["position"] += delta
                return "ACK"
            else:
                return f"NACK: Axis '{axis}' not found or not connected"
        except Exception as e:
            return f"NACK: {e}"

    elif tokens[0].lower() == "moveabs":
        if len(tokens) != 3:
            return "NACK: Usage is 'moveabs <axis> <position>'"
        axis, pos = tokens[1], tokens[2]
        try:
            pos = float(pos)
            if axis in motor_state and motor_state[axis]["is_connected"]:
                motor_state[axis]["position"] = pos
                return "ACK"
            else:
                return f"NACK: Axis '{axis}' not found or not connected"
        except Exception as e:
            return f"NACK: {e}"

    elif tokens[0].lower() == "connect":
        if len(tokens) != 2:
            return "NACK: Usage is 'connect <axis>'"
        axis = tokens[1]
        if axis in motor_state:
            motor_state[axis]["is_connected"] = True
            return "ACK"
        else:
            return f"NACK: Axis '{axis}' not found"

    elif tokens[0].lower() == "disconnect":
        if len(tokens) != 2:
            return "NACK: Usage is 'disconnect <axis>'"
        axis = tokens[1]
        if axis in motor_state:
            motor_state[axis]["is_connected"] = False
            return "ACK"
        else:
            return f"NACK: Axis '{axis}' not found"

    elif tokens[0].lower() == "state":
        if len(tokens) != 2:
            return "NACK: Usage is 'state <axis>'"
        axis = tokens[1]
        if axis in motor_state:
            return "connected" if motor_state[axis]["is_connected"] else "disconnected"
        else:
            return f"NACK: Axis '{axis}' not found"
    

    elif tokens[0].lower() == "fpm_move":
        # Usage: fpm_move <beam> <mask_name>
        if len(tokens) != 3:
            return "ERR Usage: fpm_move <beam> <mask_name>"
        try:
            beam = int(tokens[1])
            mask_name = tokens[2].upper()

            if beam not in phasemask_positions:
                return f"ERR beam {beam} not in known phasemask positions"

            x, y = phasemask_positions[beam][mask_name]
            motor_state[f"BMX{beam}"]["position"] = x
            motor_state[f"BMY{beam}"]["position"] = y
            return f"ACK moved beam {beam} to mask {mask_name} at ({x:.2f}, {y:.2f})"
        except KeyError:
            return f"ERR mask '{mask_name}' or beam '{beam}' not found"
        except Exception as e:
            return f"ERR: {e}"

    elif tokens[0].lower() == "fpm_whereami":
        # Usage: fpm_whereami <beam>
        if len(tokens) != 2:
            return "ERR Usage: fpm_whereami <beam>"
        try:
            beam = int(tokens[1])
            x_now = motor_state[f"BMX{beam}"]["position"]
            y_now = motor_state[f"BMY{beam}"]["position"]
            tol = 2.0

            closest_mask = None
            min_dist = float("inf")
            for mask, (x_ref, y_ref) in phasemask_positions.get(beam, {}).items():
                dist = ((x_ref - x_now) ** 2 + (y_ref - y_now) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_mask = mask

            if min_dist < tol:
                return closest_mask
            else:
                return "" #"NONE"
        except Exception as e:
            return f"ERR: {e}"
    
    else:
        return f"NACK: Unknown or unsupported command: {cmd}"

# def handle_mds_command(cmd: str) -> str:
#     cmd = cmd.strip().lower()
#     if cmd == "on sbb":
#         mds_state["SBB"] = True
#         return "OK"
#     elif cmd == "off sbb":
#         mds_state["SBB"] = False
#         return "OK"
#     else:
#         return "NOK"
    


# --- Camera ZMQ Thread ---
def camera_server():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:6667")
    print("[FakeCameraServer] Listening on tcp://127.0.0.1:6667")
    while True:
        try:
            msg = socket.recv_string()
            print(f"[CameraZMQ] Received: {msg}")
            reply = handle_camera_command(msg)
            socket.send_string(f'"{reply}\\r\\n"')
        except Exception as e:
            print(f"[Camera ERROR] {e}")
            socket.send_string('"Error\\r\\n"')

# --- MDS ZMQ Thread ---
def mds_server():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    print("[FakeMDSServer] Listening on tcp://127.0.0.1:5555")
    while True:
        try:
            msg = socket.recv_string()
            print(f"[MDSZMQ] Received: {msg}")
            reply = handle_mds_command(msg)
            socket.send_string(reply)
        except Exception as e:
            print(f"[MDS ERROR] {e}")
            socket.send_string("NOK")

# --- Run both servers ---
if __name__ == "__main__":
    threading.Thread(target=camera_server, daemon=True).start()
    threading.Thread(target=mds_server, daemon=True).start()

    print("[Simulation] Fake Camera + MDS server running...")
    while True:
        time.sleep(1)


""" 
# client to test eg.
import zmq

# --- Test Camera Server ---
ctx = zmq.Context()
cam_socket = ctx.socket(zmq.REQ)
cam_socket.connect("tcp://127.0.0.1:6667")
cam_socket.send_string('cli "gain"')
print("Camera reply:", cam_socket.recv_string())

# --- Test MDS Server ---
mds_socket = ctx.socket(zmq.REQ)
mds_socket.connect("tcp://127.0.0.1:5555")
mds_socket.send_string("on SBB")
print("MDS reply (on):", mds_socket.recv_string())
mds_socket.send_string("off SBB")
print("MDS reply (off):", mds_socket.recv_string())

"""