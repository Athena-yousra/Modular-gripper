import asyncio
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from asyncua import Client

async def main():
    model = mujoco.MjModel.from_xml_path("main.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=360, width=480)
    
    arm_actuator_ids = [model.actuator(f"actuator{i}").id for i in range(1, 8)]
    gripper_actuator_id = model.actuator("motor_pos_ctrl").id

    viewer = mujoco.viewer.launch_passive(model, data)

    async with Client(url="opc.tcp://127.0.0.1:4840/freeopcua/server/") as client:
        idx = await client.get_namespace_index("http://kuka.gripper.system")
        robot_obj = await client.nodes.objects.get_child([f"{idx}:RobotSystem"])

        n_arm_target = await robot_obj.get_child([f"{idx}:ArmTargetCtrl"])
        n_gripper_target = await robot_obj.get_child([f"{idx}:GripperTargetCtrl"])
        n_arm_qpos = await robot_obj.get_child([f"{idx}:ArmActualQpos"])
        n_us = await robot_obj.get_child([f"{idx}:UltrasonicDistance"])
        n_force_l = await robot_obj.get_child([f"{idx}:LeftForce"])
        n_force_r = await robot_obj.get_child([f"{idx}:RightForce"])
        n_avg_force = await robot_obj.get_child([f"{idx}:AvgForce"])
        n_cam = await robot_obj.get_child([f"{idx}:CameraFrame"])

        # Sensor fallback logic matching main.py
        try:
            sensor_us = data.sensor("gripper_ultrasonic")
        except KeyError:
            sensor_us = data.sensor("ultrasonic_sensor")

        try:
            sensor_fl = data.sensor("left_finger_force")
            sensor_fr = data.sensor("right_finger_force")
        except KeyError:
            sensor_fl = data.sensor("left_pressure")
            sensor_fr = data.sensor("right_pressure")

        step_count = 0
        print("Simulation Node running with active force measurement...")
        
        while viewer.is_running():
            arm_ctrl = await n_arm_target.get_value()
            gripper_ctrl = await n_gripper_target.get_value()

            data.ctrl[arm_actuator_ids] = arm_ctrl
            data.ctrl[gripper_actuator_id] = gripper_ctrl

            mujoco.mj_step(model, data)
            viewer.sync()

            # Vector norm calculation matching main.py get_current_forces()
            fl = float(np.linalg.norm(sensor_fl.data))
            fr = float(np.linalg.norm(sensor_fr.data))
            us = float(sensor_us.data[0])

            await n_arm_qpos.set_value(data.qpos[:7].tolist())
            await n_us.set_value(us)
            await n_force_l.set_value(fl)
            await n_force_r.set_value(fr)
            await n_avg_force.set_value((fl + fr) / 2.0)

            if step_count % 16 == 0:
                renderer.update_scene(data, camera="gripper_camera")
                rgb = renderer.render()
                _, encoded_img = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                await n_cam.set_value(encoded_img.tobytes())

            step_count += 1
            await asyncio.sleep(model.opt.timestep)

    viewer.close()

if __name__ == "__main__":
    asyncio.run(main())
