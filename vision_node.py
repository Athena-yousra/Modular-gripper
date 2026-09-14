import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from asyncua import Client

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.8
TARGET_FORCE = 4.5  # Newtons

CAM_W, CAM_H = 480, 360
FOVY = np.radians(60.0)

def ray_cast_to_table(u, v, table_z=0.43):
    fy = (CAM_H / 2.0) / np.tan(FOVY / 2.0)
    fx = fy
    cx, cy = CAM_W / 2.0, CAM_H / 2.0

    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    ray_dir_cam = np.array([x_c, y_c, 1.0])
    ray_dir_cam /= np.linalg.norm(ray_dir_cam)

    # Fixed camera pose transformation from gripper site to world table frame
    cam_pos_world = np.array([0.70, -0.01, 0.65])
    
    if ray_dir_cam[2] != 0:
        t = (table_z - cam_pos_world[2]) / ray_dir_cam[2]
        world_pt = cam_pos_world + t * ray_dir_cam
        return world_pt
    return np.array([0.70, -0.01, table_z])

def get_zone_info(speed_factor):
    if speed_factor <= 0.0:
        return "RED (STOP)", (0, 0, 255)         # Red
    elif speed_factor <= 0.15:
        return "ORANGE (15%)", (0, 165, 255)     # Orange
    elif speed_factor <= 0.40:
        return "YELLOW (40%)", (0, 255, 255)     # Yellow
    else:
        return "GREEN (100%)", (0, 255, 0)       # Green

async def main():
    model = YOLO(MODEL_PATH)
    cv2.namedWindow("YOLO Vision & Telemetry HUD", cv2.WINDOW_NORMAL)

    async with Client(url="opc.tcp://127.0.0.1:4840/freeopcua/server/") as client:
        idx = await client.get_namespace_index("http://kuka.gripper.system")
        robot_obj = await client.nodes.objects.get_child([f"{idx}:RobotSystem"])

        # Fetch OPC UA telemetry nodes
        n_cam = await robot_obj.get_child([f"{idx}:CameraFrame"])
        n_us = await robot_obj.get_child([f"{idx}:UltrasonicDistance"])
        n_force_l = await robot_obj.get_child([f"{idx}:LeftForce"])
        n_force_r = await robot_obj.get_child([f"{idx}:RightForce"])
        n_avg_force = await robot_obj.get_child([f"{idx}:AvgForce"])
        n_speed = await robot_obj.get_child([f"{idx}:SpeedFactor"])

        print("Vision YOLO & Telemetry HUD Node Running...")

        while True:
            frame_bytes = await n_cam.get_value()
            if len(frame_bytes) > 0:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # 1. Fetch live telemetry data from OPC UA
                    us_dist = await n_us.get_value()
                    f_left = await n_force_l.get_value()
                    f_right = await n_force_r.get_value()
                    f_avg = await n_avg_force.get_value()
                    speed_factor = await n_speed.get_value()

                    # Format ultrasonic text and safety zone information
                    if us_dist < 0 or us_dist >= 4.0:
                        us_text = "US Dist: Out of Range"
                    else:
                        us_text = f"US Dist: {us_dist:.3f} m"

                    current_zone, zone_color = get_zone_info(speed_factor)

                    # 2. Perform YOLO inference and raycasting
                    results = model(img, conf=CONF_THRESHOLD, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            u_center = (x1 + x2) / 2.0
                            v_center = (y1 + y2) / 2.0

                            world_pos = ray_cast_to_table(u_center, v_center)

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"Lego: [{world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}]"
                            cv2.putText(img, text, (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                    # 3. Build HUD overlay lines list
                    lines = [
                        (us_text, (0, 255, 0)),
                        (f"Force L: {f_left:.2f} N", (0, 255, 255)),
                        (f"Force R: {f_right:.2f} N", (0, 255, 255)),
                        (f"Avg Force: {f_avg:.2f} / {TARGET_FORCE:.1f} N", (255, 255, 0)),
                        (f"ZONE: {current_zone}", zone_color)
                    ]

                    # 4. Render HUD lines on top-left of the camera view
                    y_offset = 25
                    for text_str, color in lines:
                        # Black background outline for high contrast
                        cv2.putText(img, text_str, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                        # Foreground colored text
                        cv2.putText(img, text_str, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                        y_offset += 22

                    cv2.imshow("YOLO Vision & Telemetry HUD", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            await asyncio.sleep(0.03)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
