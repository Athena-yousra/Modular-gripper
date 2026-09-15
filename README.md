# KUKA iiwa Pick-and-Place — MuJoCo Simulation

A KUKA LBR iiwa 14 arm fitted with a custom two-finger gripper (RB9),
simulated in MuJoCo, that uses a wrist-mounted camera + YOLO to detect a
lego piece and moves the arm through the pick-and-place motion using a
frame-to-frame joint interpolation approach between fixed waypoints,
gripping the object with closed-loop force control. Four processes
(Simulation, Vision, Control, and an OPC UA server) coordinate over the
OPC UA industrial protocol, so each part can be developed, tested, or
replaced independently.

For what each file does internally, see
[`CODE_OVERVIEW.md`](CODE_OVERVIEW.md).

## Before you start
```
modular-gripper/
├── assets/              ← every mesh file, flat, no subfolders:
│                            link_0.obj ... link_7.obj, band.obj, kuka.obj  (arm, 8 files)
│                            01_..._1.stl ... 23_..._Hex.stl                (gripper, 23 files)
│                            lego.stl                                       (lego piece, 1 file)
├── main.xml
├── kuka_arm.xml
├── ...
```

All mesh file paths in `main.xml` and `gripper_hardware.xml` are already
**relative**  resolved against a single
`meshdir="assets"`. So once all 32 files are sitting flat in `assets/`,
everything resolves automatically — no path to edit, on any machine, as
long as you run the scripts **from inside this folder**.

## Setup

### 1. Get the project onto your machine

```
mkdir modular-gripper
cd modular-gripper
```

Put every file from this repo directly inside that folder

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
5. **UI** — runs the scripted robot.ui
 ```
   python robot_ui.py
   ```
To stop everything, `Ctrl+C` each terminal (stop the OPC server last).

 
- Safety zones (`SpeedFactor`) scale the arm's motion speed based on the
  live ultrasonic/rangefinder distance, following a simplified Speed and
  Separation Monitoring (SSM) approach.

