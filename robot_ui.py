#!/usr/bin/env python3
"""
robot_ui.py — All-in-one Tkinter control panel for the Modular-gripper
MuJoCo project (KUKA iiwa 14 + RB9 gripper + LEGO piece).

Tabs / tasks
------------
1. Joints    Manual control of the 7 arm actuators with sliders
2. Gripper   Open / close the two-finger gripper + live force feedback
3. Vision    Wrist-camera feed, YOLO detection and world coordinates
             (display only — the robot does NOT move in this tab)
4. Pick&Pl   Full pick-and-place sequence with force-controlled gripping

Toolbar: Run / Pause / Continue / Reset for the simulation.

Run from inside the project folder (where main.xml lives):

    python robot_ui.py

No OPC UA needed — this panel talks to MuJoCo directly.
Deps: mujoco, numpy, opencv-python. YOLO needs ultralytics (optional).
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import cv2
import mujoco
import mujoco.viewer

# ----------------------------------------------------------------------------
# Constants (mirrors control_node.py)
# ----------------------------------------------------------------------------
HOME_POS      = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PRE_PICK_POS  = [0.0, 0.147, 0.0, -0.817, 0.0, 1.28, -1.59]
PICK_POS      = [0.0, 0.733, 0.0, -0.837, 0.0, 1.28, -1.59]
PRE_PLACE_POS = [-1.2, 0.147, 0.0, -0.817, 0.0, 1.28, -1.59]
PLACE_POS     = [-1.2, 0.733, 0.0, -0.837, 0.0, 1.28, -1.59]

# Pose the "home" keyframe in main.xml puts the arm into (used by Reset / Go Home)
KEYFRAME_HOME = [0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0]

GRIPPER_OPEN = 7.29          # motor_joint value with fingers fully open
TARGET_FORCE = 4.5           # Newtons, from control_node.py
CAMERA_NAME  = "gripper_camera"
EE_SITE      = "attachment_site"
LEGO_BODY    = "lego_piece"

CAM_W, CAM_H = 480, 360
FOVY = np.radians(60.0)


# ----------------------------------------------------------------------------
# Shared state between physics thread, sequence thread, YOLO thread and UI
# ----------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = False                 # set on UI close -> everything shuts down
        self.run_event = threading.Event()
        self.run_event.set()              # cleared = simulation paused

        self.reset_requested = False
        self.arm_target = np.array(HOME_POS, dtype=float)
        self.gripper_target = GRIPPER_OPEN

        # telemetry (written by physics thread)
        self.arm_qpos = np.zeros(7)
        self.ee_pos = np.zeros(3)
        self.lego_pos = np.zeros(3)
        self.force_l = 0.0
        self.force_r = 0.0
        self.avg_force = 0.0
        self.us_dist = -1.0
        self.camera_rgb = None            # latest wrist-camera frame (H,W,3) RGB
        self.frame_id = 0

        # pick & place
        self.seq_running = False
        self.abort_seq = False
        self.seq_finished = False         # UI consumes this to re-sync sliders
        self.phase_text = "idle"

        # vision
        self.yolo_enabled = False


# ----------------------------------------------------------------------------
# Pick-and-place sequence (runs in its own thread, writes joint targets)
# ----------------------------------------------------------------------------
def run_pick_place(state: SharedState):
    grip_lock = GRIPPER_OPEN

    def set_targets(arm=None, grip=None, phase=None):
        with state.lock:
            if arm is not None:
                state.arm_target[:] = arm
            if grip is not None:
                state.gripper_target = grip
            if phase is not None:
                state.phase_text = phase

    def move_to(start, target, mode, duration, label):
        nonlocal grip_lock
        t0 = time.time()
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        while True:
            if state.abort_seq or state.stop:
                return False
            state.run_event.wait()                     # hold still while paused
            if state.abort_seq:
                return False
            now = time.time()
            p = min(1.0, (now - t0) / max(duration, 0.05))
            s = p * p * (3.0 - 2.0 * p)                # smoothstep ease
            arm = (1.0 - s) * start + s * target

            with state.lock:
                f = state.avg_force
            if mode in ("OPEN", "OPEN_RELEASE"):
                grip_lock = GRIPPER_OPEN
                grip = GRIPPER_OPEN
            elif mode == "FORCE_GRIP":
                if f < TARGET_FORCE and grip_lock > 0.0:
                    grip_lock = max(0.0, grip_lock - 0.035)   # 3.5 / 0.01 s
                grip = grip_lock
            else:  # HOLD
                grip = grip_lock

            set_targets(arm=arm, grip=grip,
                        phase=f"{label} | grip={mode} | p={p*100:3.0f}%")
            if p >= 1.0:
                return True
            time.sleep(0.01)

    state.seq_running = True
    state.abort_seq = False
    ok = True

    # Same legs as control_node.py: approach -> descend -> force grip -> lift
    # -> transfer -> descend -> release -> retreat
    legs = [
        ("Approach above object", PRE_PICK_POS,  "OPEN",         2.0),
        ("Descend onto object",   PICK_POS,      "OPEN",         2.5),
        ("Close until 4.5 N",     PICK_POS,      "FORCE_GRIP",   2.0),
        ("Lift clear",            PRE_PICK_POS,  "HOLD",         2.0),
        ("Transfer to place",     PRE_PLACE_POS, "HOLD",         3.0),
        ("Descend to place",      PLACE_POS,     "HOLD",         2.0),
        ("Release object",        PLACE_POS,     "OPEN_RELEASE", 1.0),
        ("Retreat",               PRE_PLACE_POS, "OPEN",         1.5),
    ]

    with state.lock:
        current = state.arm_target.copy()

    ok = move_to(current, HOME_POS, "OPEN", 2.0, "Move home")
    for label, target, mode, dur in legs:
        if not ok:
            break
        with state.lock:
            current = state.arm_target.copy()
        ok = move_to(current, target, mode, dur, label)

    with state.lock:
        state.seq_running = False
        state.seq_finished = True
        state.phase_text = "finished" if ok and not state.abort_seq else "aborted"


# ----------------------------------------------------------------------------
# Physics + passive viewer (main thread — GLFW wants the main thread)
# ----------------------------------------------------------------------------
def physics_main(state: SharedState, xml_path: str):
    if not os.path.exists(xml_path):
        print(f"[robot_ui] ERROR: '{xml_path}' not found.\n"
              f"[robot_ui] Run this script from inside the project folder "
              f"(where main.xml lives).")
        state.stop = True
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
               for i in range(1, 8)]
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_pos_ctrl")

    def sensor_id(names):
        for n in names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            if sid >= 0:
                return sid
        return -1

    s_us = sensor_id(["gripper_ultrasonic", "ultrasonic_sensor"])
    s_fl = sensor_id(["left_finger_force", "left_pressure"])
    s_fr = sensor_id(["right_finger_force", "right_pressure"])

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    lego_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEGO_BODY)
    home_key = model.key("home").id if "home" in [model.key(i).name for i in range(model.nkey)] else -1

    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    viewer = mujoco.viewer.launch_passive(model, data)
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        with state.lock:
            state.arm_target[:] = KEYFRAME_HOME

    step = 0
    while not state.stop and viewer.is_running():
        with state.lock:
            if state.reset_requested:
                if home_key >= 0:
                    mujoco.mj_resetDataKeyframe(model, data, home_key)
                    state.arm_target[:] = KEYFRAME_HOME
                    state.gripper_target = GRIPPER_OPEN
                mujoco.mj_forward(model, data)
                state.reset_requested = False
                state.seq_finished = True     # makes the UI re-sync its sliders

            if state.run_event.is_set():
                data.ctrl[arm_ids] = state.arm_target
                data.ctrl[grip_id] = state.gripper_target
                mujoco.mj_step(model, data)
                step += 1
                if step % 8 == 0:
                    renderer.update_scene(data, camera=CAMERA_NAME)
                    state.camera_rgb = renderer.render().copy()
                    state.frame_id += 1

            state.arm_qpos = data.qpos[:7].copy()
            if ee_id >= 0:
                state.ee_pos = data.site(ee_id).xpos.copy()
            if lego_id >= 0:
                state.lego_pos = data.body(lego_id).xpos.copy()
            state.force_l = float(np.linalg.norm(data.sensordata[s_fl])) if s_fl >= 0 else 0.0
            state.force_r = float(np.linalg.norm(data.sensordata[s_fr])) if s_fr >= 0 else 0.0
            state.avg_force = (state.force_l + state.force_r) / 2.0
            state.us_dist = float(data.sensordata[s_us]) if s_us >= 0 else -1.0

        viewer.sync()
        time.sleep(model.opt.timestep)

    state.stop = True
    try:
        viewer.close()
    except Exception:
        pass
    renderer.close()


# ----------------------------------------------------------------------------
# YOLO worker (optional) — inference off the UI thread
# ----------------------------------------------------------------------------
def yolo_worker(state: SharedState, model_path="yolov8n.pt", conf=0.8):
    try:
        from ultralytics import YOLO
    except Exception:
        return
    if not os.path.exists(model_path):
        return
    try:
        model = YOLO(model_path)
    except Exception:
        return

    def ray_cast_to_table(u, v, table_z=0.43):
        fy = (CAM_H / 2.0) / np.tan(FOVY / 2.0)
        fx, cx, cy = fy, CAM_W / 2.0, CAM_H / 2.0
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        ray /= np.linalg.norm(ray)
        cam_pos = np.array([0.70, -0.01, 0.65])       # fixed assumed pose (as in vision_node.py)
        t = (table_z - cam_pos[2]) / ray[2]
        return cam_pos + t * ray

    last_id = -1
    while not state.stop:
        if not state.yolo_enabled:
            time.sleep(0.2)
            continue
        with state.lock:
            rgb = None if state.camera_rgb is None else state.camera_rgb.copy()
            fid = state.frame_id
        if rgb is None or fid == last_id:
            time.sleep(0.02)
            continue
        last_id = fid

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dets = []
        try:
            for r in model(img, conf=conf, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    wp = ray_cast_to_table((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    dets.append((x1, y1, x2, y2, wp))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Lego [{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}]",
                                (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        except Exception:
            pass
        yolo_last = {"img": img, "dets": dets}
        with state.lock:
            state.yolo_last = yolo_last


# ----------------------------------------------------------------------------
# Tkinter UI (runs in its own thread)
# ----------------------------------------------------------------------------
def run_ui(state: SharedState, joint_limits):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Modular Gripper — Robot Control Panel")
    root.protocol("WM_DELETE_WINDOW", lambda: on_close())

    # ---- toolbar -----------------------------------------------------------
    bar = ttk.Frame(root, padding=6)
    bar.pack(side=tk.TOP, fill=tk.X)

    def do_reset():
        with state.lock:
            state.abort_seq = True
            state.reset_requested = True

    ttk.Button(bar, text="▶ Run / Continue", command=lambda: state.run_event.set()).pack(side=tk.LEFT, padx=3)
    ttk.Button(bar, text="⏸ Pause", command=lambda: state.run_event.clear()).pack(side=tk.LEFT, padx=3)
    ttk.Button(bar, text="⟲ Reset", command=do_reset).pack(side=tk.LEFT, padx=3)
    status_lbl = ttk.Label(bar, text="RUNNING", font=("", 10, "bold"))
    status_lbl.pack(side=tk.LEFT, padx=12)

    def on_close():
        with state.lock:
            state.stop = True
        state.run_event.set()
        try:
            root.destroy()
        except Exception:
            pass

    # ---- telemetry strip ----------------------------------------------------
    tele = ttk.LabelFrame(root, text="Live telemetry", padding=6)
    tele.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
    tele_lbl = {}
    for key in ("ee", "lego", "fl", "fr", "favg", "us", "qpos"):
        tele_lbl[key] = ttk.Label(tele, text="—", width=26, anchor="w")
    for i, key in enumerate(("ee", "lego", "fl", "fr", "favg", "us")):
        tele_lbl[key].grid(row=0, column=i, sticky="w", padx=4)
    tele_lbl["qpos"].grid(row=1, column=0, columnspan=6, sticky="w", padx=4)

    # ---- notebook with the 4 task tabs --------------------------------------
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ============================ TAB 1 · JOINTS =============================
    tab1 = ttk.Frame(nb, padding=8)
    nb.add(tab1, text=" 1 · Joint Control ")
    ttk.Label(tab1, text="Drag a slider to command that joint (the pick-and-place sequence aborts if you do).",
              foreground="#555").pack(anchor="w")
    ttk.Button(tab1, text="Go Home (keyframe)", command=lambda: go_home()).pack(anchor="w", pady=4)

    joint_sliders, joint_vals = [], []
    rows = ttk.Frame(tab1)
    rows.pack(fill=tk.X)
    for i in range(7):
        lo, hi = joint_limits[i]
        row = ttk.Frame(rows)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=f"joint{i+1}", width=8).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=KEYFRAME_HOME[i])
        lbl = ttk.Label(row, text="", width=26)
        lbl.pack(side=tk.RIGHT)
        sl = ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var,
                       command=lambda v, i=i: joint_moved(i, float(v)))
        sl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        joint_sliders.append((sl, var))
        joint_vals.append(lbl)

    def joint_moved(i, v):
        with state.lock:
            state.abort_seq = True          # manual takeover
            state.arm_target[i] = v

    def go_home():
        with state.lock:
            state.abort_seq = True
            state.arm_target[:] = KEYFRAME_HOME
        for (sl, var), v in zip(joint_sliders, KEYFRAME_HOME):
            var.set(v)

    # ============================ TAB 2 · GRIPPER ============================
    tab2 = ttk.Frame(nb, padding=8)
    nb.add(tab2, text=" 2 · Gripper ")
    ttk.Label(tab2, text="Open / close the two-finger gripper (rack-and-pinion, single motor).",
              foreground="#555").pack(anchor="w")

    gvar = tk.DoubleVar(value=GRIPPER_OPEN)
    gf = ttk.Frame(tab2)
    gf.pack(fill=tk.X, pady=6)
    ttk.Label(gf, text="motor_joint target", width=18).pack(side=tk.LEFT)
    gscale = ttk.Scale(gf, from_=0.0, to=8.0, orient=tk.HORIZONTAL, variable=gvar,
                       command=lambda v: set_grip(float(v)))
    gscale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
    gval_lbl = ttk.Label(gf, text="", width=8)
    gval_lbl.pack(side=tk.RIGHT)

    bf = ttk.Frame(tab2)
    bf.pack(anchor="w", pady=4)
    ttk.Button(bf, text="Open (7.29)", command=lambda: set_grip(GRIPPER_OPEN)).pack(side=tk.LEFT, padx=3)
    ttk.Button(bf, text="Half (3.6)",  command=lambda: set_grip(3.6)).pack(side=tk.LEFT, padx=3)
    ttk.Button(bf, text="Close (0.0)", command=lambda: set_grip(0.0)).pack(side=tk.LEFT, padx=3)

    gforce_lbl = ttk.Label(tab2, text="", font=("", 10))
    gforce_lbl.pack(anchor="w", pady=8)

    def set_grip(v):
        with state.lock:
            state.gripper_target = v

    # ============================= TAB 3 · VISION ============================
    tab3 = ttk.Frame(nb, padding=8)
    nb.add(tab3, text=" 3 · Vision (no motion) ")
    top3 = ttk.Frame(tab3)
    top3.pack(fill=tk.X)
    ttk.Label(top3, text="Wrist-camera feed + YOLO detection. DISPLAY ONLY — the robot does not move in this tab.",
              foreground="#555").pack(side=tk.LEFT)
    yolo_chk = tk.BooleanVar(value=False)
    yolo_available = os.path.exists("yolov8n.pt")
    def toggle_yolo():
        with state.lock:
            state.yolo_enabled = yolo_chk.get()
    ttk.Checkbutton(top3, text="YOLO detection (yolov8n.pt)" + ("" if yolo_available else " — model file not found"),
                    variable=yolo_chk, command=toggle_yolo,
                    state=tk.NORMAL if yolo_available else tk.DISABLED).pack(side=tk.RIGHT)

    mid3 = ttk.Frame(tab3)
    mid3.pack(fill=tk.BOTH, expand=True)
    cam_lbl = ttk.Label(mid3)
    cam_lbl.pack(side=tk.LEFT, padx=4)
    side3 = ttk.Frame(mid3)
    side3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
    ttk.Label(side3, text="Coordinates", font=("", 10, "bold")).pack(anchor="w")
    coords_txt = tk.Text(side3, width=44, height=16, font=("Consolas", 9))
    coords_txt.pack(fill=tk.BOTH, expand=True)

    # =========================== TAB 4 · PICK & PLACE ========================
    tab4 = ttk.Frame(nb, padding=8)
    nb.add(tab4, text=" 4 · Pick & Place ")
    ttk.Label(tab4, text="Runs the full sequence: approach → descend → force-grip (4.5 N) → lift → "
                         "transfer → place → release → retreat.", foreground="#555").pack(anchor="w")
    b4 = ttk.Frame(tab4)
    b4.pack(anchor="w", pady=6)
    ttk.Button(b4, text="▶ Start pick & place", command=start_seq).pack(side=tk.LEFT, padx=3)
    ttk.Button(b4, text="✖ Abort", command=lambda: abort_seq()).pack(side=tk.LEFT, padx=3)

    phase_lbl = ttk.Label(tab4, text="idle", font=("", 10, "bold"))
    phase_lbl.pack(anchor="w", pady=4)
    prog = ttk.Progressbar(tab4, length=420, mode="determinate")
    prog.pack(anchor="w", pady=4)
    steps_txt = tk.Text(tab4, width=70, height=9, font=("Consolas", 9))
    steps_txt.pack(anchor="w", pady=6)
    steps_txt.insert(tk.END, "Planned legs (from control_node.py):\n")
    for i, (lbl_, _, mode_, dur_) in enumerate([
            ("Approach above object", "", "OPEN", 2.0), ("Descend onto object", "", "OPEN", 2.5),
            ("Close until 4.5 N", "", "FORCE_GRIP", 2.0), ("Lift clear", "", "HOLD", 2.0),
            ("Transfer to place", "", "HOLD", 3.0), ("Descend to place", "", "HOLD", 2.0),
            ("Release object", "", "OPEN_RELEASE", 1.0), ("Retreat", "", "OPEN", 1.5)], 1):
        steps_txt.insert(tk.END, f"  {i}. {lbl_:26s} grip={mode_:12s} {dur_:3.1f}s\n")
    steps_txt.config(state=tk.DISABLED)

    seq_thread = [None]

    def start_seq():
        if seq_thread[0] and seq_thread[0].is_alive():
            return
        abort_seq()
        seq_thread[0] = threading.Thread(target=run_pick_place, args=(state,), daemon=True)
        seq_thread[0].start()

    def abort_seq():
        with state.lock:
            state.abort_seq = True

    # ---- periodic refresh ---------------------------------------------------
    img_ref = [None]

    def tick():
        if state.stop:
            try:
                root.destroy()
            except Exception:
                pass
            return
        with state.lock:
            q = state.arm_qpos.copy()
            ee = state.ee_pos.copy()
            lp = state.lego_pos.copy()
            fl, fr, fa = state.force_l, state.force_r, state.avg_force
            us = state.us_dist
            phase = state.phase_text
            seq_on = state.seq_running
            rgb = None if state.camera_rgb is None else state.camera_rgb.copy()
            yolo = getattr(state, "yolo_last", None)
            finished = state.seq_finished
            state.seq_finished = False
            gtarget = state.gripper_target
        running = state.run_event.is_set()

        status_lbl.config(text="RUNNING" if running else "PAUSED")
        tele_lbl["ee"].config(text=f"EE (robot): [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}]")
        tele_lbl["lego"].config(text=f"LEGO: [{lp[0]:.2f}, {lp[1]:.2f}, {lp[2]:.2f}]")
        tele_lbl["fl"].config(text=f"F left:  {fl:5.2f} N")
        tele_lbl["fr"].config(text=f"F right: {fr:5.2f} N")
        tele_lbl["favg"].config(text=f"F avg:   {fa:5.2f} / {TARGET_FORCE} N")
        tele_lbl["us"].config(text=f"US dist: {us:.3f} m" if 0 <= us < 4 else "US dist: out of range")
        tele_lbl["qpos"].config(text="qpos: " + "  ".join(f"{v:+.2f}" for v in q))

        # joints tab readouts
        for i, lbl in enumerate(joint_vals):
            lbl.config(text=f"target {at[i]:+.2f}   actual {q[i]:+.2f}")

        # gripper tab
        gval_lbl.config(text=f"{gtarget:.2f}")
        gforce_lbl.config(text=f"Left {fl:.2f} N   Right {fr:.2f} N   Avg {fa:.2f} N "
                               f"(grip target {TARGET_FORCE} N)")

        # camera feed (YOLO-annotated if enabled and available)
        show = None
        if yolo is not None and yolo_chk.get():
            show = yolo["img"]
        elif rgb is not None:
            show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if show is not None:
            small = cv2.resize(show, (480, 360))
            png = cv2.imencode(".png", small)[1].tobytes()
            img_ref[0] = tk.PhotoImage(data=png)
            cam_lbl.config(image=img_ref[0])

        # vision coordinates panel
        coords_txt.config(state=tk.NORMAL)
        coords_txt.delete("1.0", tk.END)
        coords_txt.insert(tk.END, f"ROBOT end-effector (attachment_site):\n"
                                  f"  x={ee[0]:+.3f}  y={ee[1]:+.3f}  z={ee[2]:+.3f}\n\n")
        coords_txt.insert(tk.END, f"LEGO piece (ground-truth body position):\n"
                                  f"  x={lp[0]:+.3f}  y={lp[1]:+.3f}  z={lp[2]:+.3f}\n\n")
        if yolo is not None and yolo_chk.get():
            coords_txt.insert(tk.END, f"YOLO detections ({len(yolo['dets'])}):\n")
            for j, (x1, y1, x2, y2, wp) in enumerate(yolo["dets"]):
                coords_txt.insert(tk.END,
                                  f"  #{j+1} box=({x1},{y1},{x2},{y2}) -> world "
                                  f"[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]\n")
        else:
            coords_txt.insert(tk.END, "YOLO off — showing raw camera feed only.")
        coords_txt.config(state=tk.DISABLED)

        # pick & place tab
        phase_lbl.config(text=("RUNNING: " if seq_on else "") + phase)
        if seq_on:
            try:
                pval = float(phase.split("p=")[1].split("%")[0])
            except Exception:
                pval = 0
            prog.config(mode="determinate", value=pval)
        else:
            prog.config(mode="determinate", value=0 if not seq_on else prog["value"])

        # after a sequence finished / reset: re-sync sliders to real targets
        if finished:
            with state.lock:
                at = state.arm_target.copy()
                gt = state.gripper_target
            for (sl, var), v in zip(joint_sliders, at):
                var.set(v)
            gvar.set(min(max(gt, 0.0), 8.0))

        root.after(120, tick)

    root.after(120, tick)
    root.mainloop()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Control panel for the Modular-gripper MuJoCo scene")
    ap.add_argument("--xml", default="main.xml", help="path to the MuJoCo scene (default: main.xml)")
    args = ap.parse_args()

    state = SharedState()

    # peek at the model for joint limits before the physics thread starts
    try:
        m = mujoco.MjModel.from_xml_path(args.xml)
        limits = [tuple(m.actuator_ctrlrange[i]) for i in range(7)]
        del m
    except Exception as e:
        print(f"[robot_ui] could not pre-load '{args.xml}': {e}")
        limits = [(-2.97, 2.97), (-2.09, 2.09), (-2.97, 2.97),
                  (-2.09, 2.09), (-2.97, 2.97), (-2.09, 2.09), (-3.05, 3.05)]

    threading.Thread(target=yolo_worker, args=(state,), daemon=True).start()
    ui_thread = threading.Thread(target=run_ui, args=(state, limits), daemon=True)
    ui_thread.start()

    physics_main(state, args.xml)     # main thread (GLFW requirement)
    state.stop = True
    ui_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
