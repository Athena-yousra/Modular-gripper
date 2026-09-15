# Code Overview

This document explains how the project is architected and what each file
does. For setup and run instructions, see [`README.md`](README.md).

## Architecture

Four independent processes coordinate exclusively through a central
**OPC UA server** — none of them call each other directly. This mirrors
how the system would be split across real hardware/controllers, and lets
each piece be developed, tested, or swapped independently.

```
                    ┌────────────────┐
                    │  opc_server.py  │   ← shared variable store
                    │   (OPC UA)      │
                    └──┬─────┬────┬──┘
         arm/gripper   │     │    │  telemetry (forces, distance,
         targets       │     │    │  camera frame, speed factor)
                       │     │    │
        ┌──────────────┘     │    └──────────────┐
        ▼                    ▼                    ▼
 control_node.py       sim_node.py          vision_node.py
 (hardcoded pick-      (MuJoCo physics +    (YOLO detection +
  and-place sequence,   IK, "the hardware")  live telemetry HUD,
  safety-zone logic)                          display-only)
```

Note that in this version, `vision_node.py` doesn't publish anything back
to the OPC UA server — it reads telemetry and the camera frame, runs
detection, and displays a HUD, but the detected box coordinates aren't
wired into `control_node.py`'s trajectory. The pick-and-place sequence
currently runs from fixed joint-angle waypoints rather than reacting to
what the camera sees (see the note in `README.md`).

## Files

### `opc_server.py`
logs every signal to `signal_log.csv` in the project
folder as it runs — no extra process needed, it writes a timestamped row
every 50 ms for as long as the server is up. 
every shared variable under a `RobotSystem` object:
- **Arm/gripper commands** — `ArmTargetCtrl` (7 joint targets),
  `GripperTargetCtrl`
- **Actual state** — `ArmActualQpos`, `GripperActualPos`
- **Telemetry** — `UltrasonicDistance`, `LeftForce`, `RightForce`,
  `AvgForce`, `SpeedFactor`
- **Status** — `CurrentPhase`, `ControlConnected`, `CameraFrame`
Must be running before any other node connects.

### `sim_node.py`
Loads `main.xml` into MuJoCo and steps the physics loop. Each iteration:
1. Reads `ArmTargetCtrl` / `GripperTargetCtrl` from OPC UA and applies
   them directly as actuator control targets (position servos — the arm
   has no separate IK solver in this file; joint targets arrive
   pre-computed from `control_node.py`).
2. Steps physics (`mujoco.mj_step`).
3. Reads back the finger force sensors and rangefinder, publishes
   `LeftForce`, `RightForce`, `AvgForce`, `UltrasonicDistance`, and
   `ArmActualQpos`.
4. Every 16 steps, renders the wrist camera and publishes the JPEG frame
   as `CameraFrame`.

Opens MuJoCo's interactive viewer.

### `vision_node.py`
*(Originally `vision_yolo_node.py`.)* Pulls the camera frame and
telemetry from OPC UA, runs YOLO object detection (`yolov8n.pt`,
`conf=0.8`), and for each detected box, ray-casts the pixel center to a
3D world point on the table plane (`ray_cast_to_table`) using a **fixed,
assumed camera pose** rather than the live camera transform — a
simplification worth knowing about if positions look slightly off.
Renders a live HUD over the camera feed showing ultrasonic distance,
per-finger and average force, and the current safety zone/speed factor.
Purely observational — publishes nothing back to OPC UA.

### `control_node.py`
Runs a fixed, hardcoded pick-and-place sequence as a list of
`(target_joint_angles, gripper_mode, duration)` steps:
`PREP → GRAB (open) → GRAB (force-grip) → PREP (hold) → GRAB (hold) →
GRAB (release) → PREP (open)`. For each step, `execute_trajectory`:
- Smoothly interpolates joint angles using a smoothstep ease
  ($s = 3p^2 - 2p^3$), scaled by the live safety-zone speed factor.
- Runs `evaluate_safety_zone`, which computes a stop/15%/40%/100% speed
  factor from the live ultrasonic distance and commanded velocity — a
  simplified Speed and Separation Monitoring (SSM) approach.
- Drives the gripper according to `mode`: fully open, a closed-loop
  force controller that keeps closing until `AvgForce` reaches
  `TARGET_FORCE` (4.5 N), or holding the last commanded position.

### `main.xml`
The MuJoCo scene: includes `kuka_arm.xml` and `ultrasonic_sensor.xml`,
adds a floor, a table, and two free-jointed boxes (`box_1` red,
`box_2` blue) as pick targets, plus a reference `home` keyframe.

### `kuka_arm.xml`
The KUKA LBR iiwa 14 kinematic chain (7 joints, `link_0`–`link_7`
meshes from `assets/`), joint limits/gains per joint class, and
7 position actuators. Includes `gripper_hardware.xml` at the top level
(for its shared `<default>`/`<asset>` declarations) and nests
`gripper_body.xml` inside `link7` so the gripper moves rigidly with the
end effector.

### `gripper_body.xml`
The physical gripper geometry: camera, camera-visual box, the
ultrasonic mounting site, two sliding fingers with force-sensor sites,
and the motor pinion — built from the 23 meshes defined in
`gripper_hardware.xml`.

### `gripper_hardware.xml`
Gripper-specific defaults, the 23 mesh asset definitions (now relative
paths under `gripper/meshes/`), the rack-and-pinion equality constraints
coupling both fingers to one motor joint, the gripper's position
actuator, and the two finger force sensors.

### `ultrasonic_sensor.xml`(laser)
. Defines a
rangefinder sensor (`gripper_ultrasonic`) on the `ultrasonic_site`
already present in `gripper_body.xml`.
### `camera.xml`
. Defines the visual camera sensor (gripper_camera) mounted on the
gripper to capture live image frames for the computer vision node.
### `force_sensor.xml`
.Defines the physical touch sensors (left_finger_force and right_finger_force)
mounted on the gripper fingertips to measure grip pressure and confirm a successful grasp.


### `yolov8n.pt`
Trained YOLOv8 model weights specifically fine-tuned for Lego brick identification,
enabling accurate bounding box detection for the red and blue target blocks in the scene.
