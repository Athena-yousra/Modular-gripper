import asyncio
from asyncua import Server, ua

async def main():
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")
    server.set_server_name("KUKA_Gripper_OPCUA_Server")

    uri = "http://kuka.gripper.system"
    idx = await server.register_namespace(uri)
    objects = server.nodes.objects

    # System Objects & Variables
    robot_obj = await objects.add_object(idx, "RobotSystem")

    # Arm & Gripper Controls
    arm_target = await robot_obj.add_variable(idx, "ArmTargetCtrl", [0.0]*7)
    gripper_target = await robot_obj.add_variable(idx, "GripperTargetCtrl", 7.29)
    arm_qpos = await robot_obj.add_variable(idx, "ArmActualQpos", [0.0]*7)
    gripper_qpos = await robot_obj.add_variable(idx, "GripperActualPos", 0.0)

    # Telemetry
    us_dist = await robot_obj.add_variable(idx, "UltrasonicDistance", -1.0)
    force_l = await robot_obj.add_variable(idx, "LeftForce", 0.0)
    force_r = await robot_obj.add_variable(idx, "RightForce", 0.0)
    avg_force = await robot_obj.add_variable(idx, "AvgForce", 0.0)
    speed_factor = await robot_obj.add_variable(idx, "SpeedFactor", 1.0)

    # Status & Vision
    phase = await robot_obj.add_variable(idx, "CurrentPhase", "INITIALIZING")
    ctrl_conn = await robot_obj.add_variable(idx, "ControlConnected", False)
    cam_frame = await robot_obj.add_variable(idx, "CameraFrame", b"")

    # Make variables writable by clients
    for var in [arm_target, gripper_target, arm_qpos, gripper_qpos, us_dist, 
                force_l, force_r, avg_force, speed_factor, phase, ctrl_conn, cam_frame]:
        await var.set_writable()

    print("OPC UA High Level Control Server started at opc.tcp://127.0.0.1:4840/freeopcua/server/")
    async with server:
        while True:
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
