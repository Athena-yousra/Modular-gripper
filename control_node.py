import asyncio
import time
import numpy as np
from asyncua import Client

HOME_POS      = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]       
PRE_PICK_POS  = [0.0, 0.147, 0.0, -0.817, 0.0, 1.28, -1.59]  
PICK_POS      = [0.0, 0.733, 0.0, -0.837, 0.0, 1.28, -1.59]  

PRE_PLACE_POS = [-1.2, 0.147, 0.0, -0.817, 0.0, 1.28, -1.59]  
PLACE_POS     = [-1.2, 0.733, 0.0, -0.837, 0.0, 1.28, -1.59]  

GRIPPER_OPEN = 7.29
TARGET_FORCE = 4.5
BASE_CLEARANCE, RESPONSE_TIME, ZONE_BUFFER = 0.08, 0.1, 0.06

# --- TASK SEQUENCE ---
# Each leg is (target_arm_pos, gripper_mode, nominal_duration_s). Grouped
# and commented by task so it maps onto Pick / Displacement / Place phases.
sequence = [
    # --- PICK TASK ---
    (PRE_PICK_POS, 'OPEN', 2.0),          # approach above the object
    (PICK_POS,     'OPEN', 2.5),          # descend onto it
    (PICK_POS,     'FORCE_GRIP', 2.0),    # close until TARGET_FORCE is reached
    (PRE_PICK_POS, 'HOLD', 2.0),          # lift clear, holding grip

    # --- DISPLACEMENT TASK (transfer while holding) ---
    (PRE_PLACE_POS, 'HOLD', 3.0),         # travel to above the place location

    # --- PLACE TASK ---
    (PLACE_POS,     'HOLD', 2.0),         # descend to the place location
    (PLACE_POS,     'OPEN_RELEASE', 1.0), # release the object
    (PRE_PLACE_POS, 'OPEN', 1.5),         # retreat back up
]

# Persistent gripper state across sequence steps
gripper_locked_pos = GRIPPER_OPEN

def evaluate_safety_zone(us_dist, arm_qvel):
    safe_dist = 4.0 if us_dist < 0 else us_dist
    true_vel = min(float(np.linalg.norm(arm_qvel)), 0.5)
    d_red = BASE_CLEARANCE + (true_vel * RESPONSE_TIME)-0.03
    d_orange, d_yellow = d_red + ZONE_BUFFER, d_red + (2 * ZONE_BUFFER)

    if safe_dist <= d_red: return 0.0, "RED (STOP)"
    elif safe_dist <= d_orange: return 0.15, "ORANGE (15%)"
    elif safe_dist <= d_yellow: return 0.40, "YELLOW (40%)"
    return 1.0, "GREEN (100%)"

async def execute_trajectory(nodes, start_arm, target_arm, mode, nominal_duration):
    global gripper_locked_pos
    n_arm_target, n_gripper_target, n_us, n_avg_force, n_arm_qpos, n_speed, n_phase = nodes
    progress = 0.0
    last_time = time.time()

    while progress < 1.0:
        dt = time.time() - last_time
        last_time = time.time()

        us_dist = await n_us.get_value()
        avg_force = await n_avg_force.get_value()

        speed_factor, zone_str = evaluate_safety_zone(us_dist, [0.0]*7)
        await n_speed.set_value(speed_factor)
        await n_phase.set_value(f"{mode} | Zone: {zone_str}")

        progress = min(1.0, progress + (dt * speed_factor) / nominal_duration)
        s = progress * progress * (3.0 - 2.0 * progress)
        curr_arm = (1.0 - s) * np.array(start_arm) + s * np.array(target_arm)
        await n_arm_target.set_value(curr_arm.tolist())

        # Closed-Loop Force Logic matching main.py
        if mode in ['OPEN', 'OPEN_RELEASE']:
            gripper_locked_pos = GRIPPER_OPEN
            await n_gripper_target.set_value(GRIPPER_OPEN)
        elif mode == 'FORCE_GRIP':
            if avg_force < TARGET_FORCE and gripper_locked_pos > 0.0:
                gripper_locked_pos -= dt * speed_factor * 3.5
                gripper_locked_pos = max(gripper_locked_pos, 0.0)
            await n_gripper_target.set_value(gripper_locked_pos)
        elif mode == 'HOLD':
            await n_gripper_target.set_value(gripper_locked_pos)

        await asyncio.sleep(0.01)

async def main():
    async with Client(url="opc.tcp://127.0.0.1:4840/freeopcua/server/") as client:
        idx = await client.get_namespace_index("http://kuka.gripper.system")
        robot_obj = await client.nodes.objects.get_child([f"{idx}:RobotSystem"])

        nodes = [
            await robot_obj.get_child([f"{idx}:ArmTargetCtrl"]),
            await robot_obj.get_child([f"{idx}:GripperTargetCtrl"]),
            await robot_obj.get_child([f"{idx}:UltrasonicDistance"]),
            await robot_obj.get_child([f"{idx}:AvgForce"]),
            await robot_obj.get_child([f"{idx}:ArmActualQpos"]),
            await robot_obj.get_child([f"{idx}:SpeedFactor"]),
            await robot_obj.get_child([f"{idx}:CurrentPhase"])
        ]

        n_ctrl_conn = await robot_obj.get_child([f"{idx}:ControlConnected"])
        await n_ctrl_conn.set_value(True)

        print("Control Node Connected. Moving to HOME POS...")
        await nodes[0].set_value(HOME_POS)
        await asyncio.sleep(1.0)

        print("Moving to PRE-PICK POS and holding for 2 seconds scanning...")
        await execute_trajectory(nodes, HOME_POS, PRE_PICK_POS, 'OPEN', 2.0)
        await asyncio.sleep(2.0)

        print("Starting Pick / Displacement / Place Sequence...")
        current_arm = PRE_PICK_POS
        for target_arm, mode, duration in sequence:
            await execute_trajectory(nodes, current_arm, target_arm, mode, duration)
            current_arm = target_arm

if __name__ == "__main__":
    asyncio.run(main())
