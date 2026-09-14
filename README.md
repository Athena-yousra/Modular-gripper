# KUKA iiwa Pick-and-Place — MuJoCo Simulation

A KUKA LBR iiwa 14 arm fitted with a custom two-finger gripper (RB9),
simulated in MuJoCo, that uses a wrist-mounted camera + YOLO to detect a
colored box, computes an inverse-kinematics solution to reach it, grips it
with closed-loop force control, and runs a scripted pick-and-place
sequence. Four processes (Simulation, Vision, Control, and an OPC UA
server) coordinate over the OPC UA industrial protocol, so each part can
be developed, tested, or replaced independently.

For what each file does internally, see
[`CODE_OVERVIEW.md`](CODE_OVERVIEW.md).

## Before you start — assets you need to add yourself

This repo does not include the actual 3D mesh files (they're binary CAD
exports, not something to regenerate). You need to copy two folders from
wherever you have them locally into this project's root, so the final
layout looks like:

```
modular-gripper/
├── assets/              ← arm meshes (link_0.obj ... link_7.obj, band.obj, kuka.obj)
├── gripper/
│   └── meshes/          ← the 23 gripper .stl files
├── main.xml
├── kuka_arm.xml
├── ...
```

All the mesh file paths in `main.xml` and `gripper_hardware.xml` have
already been changed to **relative** paths (they used to be hardcoded to
one machine, e.g. `/home/yousra/gripper/meshes/...`) — so once the
`assets/` and `gripper/meshes/` folders are sitting alongside the other
files, everything resolves automatically. You don't need to edit any
path in any file, on any machine, as long as you run the scripts **from


## Setup

### 1. Get the project onto your machine

```
mkdir modular-gripper
cd modular-gripper
```

Put every file from this repo directly inside that folder, then add your
`assets/` and `gripper/meshes/` folders as described above.

### 2. Create a virtual environment and install dependencies

```
python3 -m venv venv
```

Activate it:
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

Then:

```
pip install -r requirements.txt
```

This installs `mujoco`, `numpy`, `opencv-python`, `asyncua` (the OPC UA
client/server library used here — note this project uses `asyncua`, not
the `opcua` package), and `ultralytics` for YOLO.

## Running it

Open **four separate terminals** (venv activated, all inside the project
folder) and start the processes **in this order**, a couple of seconds
apart:

1. **OPC UA server** — must be running before anything else connects:
   ```
   python opc_server.py
   ```
2. **Simulation node** — loads `main.xml`, opens the MuJoCo viewer:
   ```
   python sim_node.py
   ```
3. **Vision node** — opens a window with the camera feed, YOLO detection
   boxes, and a live telemetry HUD (distance, forces, safety zone):
   ```
   python vision_node.py
   ```
4. **Control node** — runs the scripted pick-and-place sequence
   (prep → approach → force-controlled grip → lift/hold → release):
   ```
   python control_node.py
   ```

To stop everything, `Ctrl+C` each terminal (stop the OPC server last).

## Notes

- The control sequence in `control_node.py` is currently **hardcoded**
  (fixed joint-angle waypoints, driven by the vision node's detected
  coordinates) — the vision pipeline runs and displays live detections,
   the arm's trajectory  reacts to what it sees. Wiring the
  detected box coordinates into the control sequence 
- Safety zones (`SpeedFactor`) scale the arm's motion speed based on the
  live ultrasonic/rangefinder distance, following a simplified Speed and
  Separation Monitoring (SSM) approach.

