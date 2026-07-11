#!/usr/bin/env python3
"""
红色飞盘检测 + 机械臂持续跟随

流程：
    1. 机械臂移动到初始关节角 [0, 0.785, 0.785, 0.785, 0, 0]
    2. 相机实时检测红色飞盘（正面/侧面）
    3. 按住 'g' 键：持续跟随物体（末端保持在物体基坐标系 X 轴向后 15cm）
    4. 松开 'g' 键：停止跟随，保持当前位置
    5. 按住空格键：闭合夹爪；松开空格键：张开夹爪              <-- 【新增】
    6. 按 'q' 退出

控制模式：MIT 模式（pos_vel_tqe_kp_kd + 重力补偿）

=====================================================================
本版本相对原始 red_frisbee_follow.py 的改动
=====================================================================
新增了一个共享状态 space_pressed（跟 g_pressed 是同样的模式），以及
control_loop() 里对应的边沿触发逻辑。改动分三处：

1. 共享状态区：新增 space_pressed[0]（当前是否按住空格）和
   last_space_state[0]（control_loop 里上一次真正触发夹爪动作时的状态，
   用来做边沿检测，避免每个 100Hz tick 都重复调用夹爪函数）。

2. on_press / on_release：只负责如实记录“当前空格是否被按住”这一个
   布尔状态，不直接调用 robot.gripper_*()。这一步只是单纯赋值，就算
   操作系统按键自动重复导致 on_press 被连续调用多次，反复赋同一个值
   也没有副作用，不需要额外处理。

   之所以不能像最初那版一样直接在这两个回调里调用 robot.gripper_close()/
   gripper_open()：pynput 的键盘监听是独立的一个线程，如果在这里直接
   调用会跟 control_loop() 线程（正在以 100Hz 持续调用
   pos_vel_tqe_kp_kd 往同一条通信链路写手臂指令）产生竞争——两个线程
   同时操作同一个 robot 对象/同一条通信总线，指令可能相互打断、覆盖，
   出现“函数调用本身没报错、print 也正常，但硬件没有真正响应”的情况。
   这是竞争条件，不是异常，try/except 对此无效，只有让所有真正对 robot
   下发指令的调用都收敛到同一个线程（control_loop）才能从根上避免。

3. control_loop()：每个 tick 读一次 space_pressed[0]，和上次触发时记录
   的 last_space_state[0] 比较，只有状态发生跳变（False->True 或
   True->False）时才真正调用一次 gripper_close()/gripper_open()，
   调用外面包了 try/except——这是另一层独立的防御，针对的是“硬件通信
   本身偶发报错”这种情况，跟前面竞争条件的问题不是一回事，两者不冲突、
   不能互相替代，所以都保留。
=====================================================================
"""

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from pynput import keyboard as pynput_kb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from red_disc_detection import (
    CircleTracker, detect_red_disc, get_median_depth, pixel_to_3d, draw_results
)
from Panthera_lib import Panthera

# ─── 配置参数 ────────────────────────────────────────────────────
HOME_JOINT_POS = [0.0, 0.785, 0.785, 0.0, 0.0, 0.0]
MAX_TORQUE     = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
JOINT_VEL      = [0.5] * 6
KP = [12.0, 24.0, 27.0, 12.0, 9.0, 6.0]
KD = [2.5,  5.0,  5.5,  2.5,  2.0,  1.3]

X_OFFSET       = -0.15   # 基坐标系 X 轴向后 15cm
Z_OFFSET       = 0.03   # 基坐标系 Z 轴向上 5cm
CTRL_RATE      = 0.01    # 控制循环周期 100Hz

# ─── 加载手眼标定 ────────────────────────────────────────────────
_CALIB_FILE = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.json")
with open(_CALIB_FILE, 'r') as _f:
    _calib = json.load(_f)
T_tcp_camera = np.array(_calib["T_tcp_camera"])

# ─── 相机初始化 ──────────────────────────────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30


def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    MIN_DEPTH_MM, MAX_DEPTH_MM = 70, 1000
    min_depth_units = MIN_DEPTH_MM / 1000.0 / depth_scale
    max_depth_units = MAX_DEPTH_MM / 1000.0 / depth_scale

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsic = color_profile.get_intrinsics()
    align = rs.align(rs.stream.color)

    print(f"Depth scale: {depth_scale} m/unit")
    print(f"相机内参: fx={intrinsic.fx:.2f} fy={intrinsic.fy:.2f} "
          f"cx={intrinsic.ppx:.2f} cy={intrinsic.ppy:.2f}")

    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, intrinsic, depth_scale, min_depth_units, max_depth_units


def camera_point_to_base(point_camera, robot):
    """相机坐标系 -> 基坐标系"""
    fk = robot.forward_kinematics()
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = np.array(fk['rotation'])
    T_base_tcp[:3, 3]  = np.array(fk['position'])
    T_base_camera = T_base_tcp @ T_tcp_camera
    p_homo = np.append(np.array(point_camera), 1.0)
    return (T_base_camera @ p_homo)[:3]


def main():
    # ── 初始化相机 ───────────────────────────────────────────────
    print("初始化 RealSense D405 相机...")
    pipeline, align, intrinsic, depth_scale, min_depth_units, max_depth_units = init_realsense()
    tracker = CircleTracker()

    # ── 初始化机械臂 ─────────────────────────────────────────────
    print("\n初始化机械臂...")
    robot = Panthera()

    print(f"\n移动到初始位置: {HOME_JOINT_POS}")
    robot.Joint_Pos_Vel(HOME_JOINT_POS, JOINT_VEL, MAX_TORQUE, iswait=True)
    time.sleep(0.5)

    fk_home = robot.forward_kinematics()
    home_rotation = np.array(fk_home['rotation'])
    print(f"初始末端位置: {np.array(fk_home['position']).round(3)} m")

    # ── 共享状态 ─────────────────────────────────────────────────
    lock = threading.Lock()
    nearest_cam = [None]      # 最近目标相机坐标 [x,y,z]
    g_pressed   = [False]     # 是否按住 g
    last_joint  = list(robot.get_current_pos())

    # ─────────────────────【新增】─────────────────────
    space_key_down    = [False]  # 物理按键当前是否处于按下状态（只在键盘线程内读写，用来过滤系统自动重复）
    toggle_pending    = [False]  # 有一次新的按下待处理（跨线程共享，需要 lock）
    gripper_closed    = [False]  # 当前夹爪的目标状态：True=闭合，False=张开
    # ───────────────────────────────────────────────

    print("\n" + "="*60)
    print("操作说明：")
    print("  按住 g: 持续跟随物体（X 轴向后 15cm）")
    print("  松开 g: 停止跟随，保持当前位置")
    print("  按住空格: 闭合夹爪；松开空格: 张开夹爪")               # 【新增】
    print("  q: 退出程序")
    print("="*60 + "\n")

    # ── pynput 键盘监听（检测 g 按住/松开、空格按住/松开）───────
    def on_press(key):
        try:
            if key.char == 'g':
                with lock:
                    g_pressed[0] = True
        except AttributeError:
            pass

        # ─────────────────────【新增】─────────────────────
        if key == pynput_kb.Key.space:
            if not space_key_down[0]:
                space_key_down[0] = True
                with lock:
                    toggle_pending[0] = True
        # ───────────────────────────────────────────────

    def on_release(key):
        try:
            if key.char == 'g':
                with lock:
                    g_pressed[0] = False
        except AttributeError:
            pass

        # ─────────────────────【新增】─────────────────────
        if key == pynput_kb.Key.space:
            space_key_down[0] = False
        # ───────────────────────────────────────────────

        if key == pynput_kb.Key.esc:
            return False

    kb_listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    # ── 控制线程 ─────────────────────────────────────────────────
    stop_ctrl = threading.Event()
    smoothed_target = [None]   # 指数平滑后的目标位置
    last_ik_target  = [None]   # 上次实际触发 IK 的目标位置
    SMOOTH_ALPHA = 0.15        # 越小越平滑，越大越灵敏
    IK_DEADZONE  = 0.010       # 目标位移死区（m），小于此值不重新求 IK

    def control_loop():
        nonlocal last_joint
        _z6 = [0.0] * robot.motor_count

        while not stop_ctrl.is_set():
            t0 = time.time()

            with lock:
                following = g_pressed[0]
                p_cam = nearest_cam[0] # 相机进程，飞盘在相机坐标系下的三维坐标
                
                pending = toggle_pending[0]
                if pending:
                    toggle_pending[0] = False

            if pending:
                gripper_closed[0] = not gripper_closed[0]
                try:
                    if gripper_closed[0]:
                        robot.gripper_close(vel=3.0)
                        print("夹爪已闭合")
                    else:
                        robot.gripper_open(vel=3.0)
                        print("夹爪已张开")
                except Exception as e:
                    print(f"夹爪动作失败: {e}")
            # ───────────────────────────────────────────────

            if following and p_cam is not None:
                # 计算目标位置
                p_base = camera_point_to_base(p_cam, robot) # 相机-机械臂坐标转换，确定飞盘在机械臂坐标下的三维坐标
                target_pos = p_base.copy()
                target_pos[0] += X_OFFSET
                target_pos[2] += Z_OFFSET

                # 指数平滑目标位置，抑制深度噪声抖动（目前不清楚逻辑）
                if smoothed_target[0] is None:
                    smoothed_target[0] = target_pos.copy()
                else:
                    smoothed_target[0] = (SMOOTH_ALPHA * target_pos
                                          + (1 - SMOOTH_ALPHA) * smoothed_target[0])

                # 死区判断：目标位移超过阈值才重新求 IK（减少运算量，在小范围内晃动不会引发机械臂运动）
                need_ik = (last_ik_target[0] is None or
                           np.linalg.norm(smoothed_target[0] - last_ik_target[0]) > IK_DEADZONE)

                if need_ik:
                    joint_pos = robot.inverse_kinematics(
                        smoothed_target[0].tolist(),
                        home_rotation,
                        last_joint,
                        multi_init=False
                    )

                    if joint_pos is not None:
                        last_joint = joint_pos
                        last_ik_target[0] = smoothed_target[0].copy()

                # 持续发送当前目标关节角（无论是否重新求解）
                robot_gra = robot.get_Gravity()
                robot_torque = np.array(robot_gra)
                robot.pos_vel_tqe_kp_kd(last_joint, _z6, robot_torque, KP, KD)

            else:
                smoothed_target[0] = None  # 松开 g 时重置平滑和死区状态
                last_ik_target[0]  = None
                # 未跟随：持续发送当前位置保持稳定
                robot_gra = robot.get_Gravity()
                robot_torque = np.array(robot_gra)
                robot.pos_vel_tqe_kp_kd(last_joint, _z6, robot_torque, KP, KD)

            elapsed = time.time() - t0
            remaining = CTRL_RATE - elapsed
            if remaining > 0:
                time.sleep(remaining)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # ── 相机主循环 ───────────────────────────────────────────────
    cv2.namedWindow('Red Frisbee Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Red Frisbee Tracking', 640, 480)
    show_mask = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            detections, mask = detect_red_disc(color_image)
            tracks = tracker.update(detections)

            for t in tracks:
                if t['lost'] == 0:
                    depth_m = get_median_depth(depth_frame, t['cx'], t['cy'],
                                               depth_scale, min_depth_units, max_depth_units)
                    if depth_m > 0:
                        t['x3d'], t['y3d'], t['z3d'] = pixel_to_3d(
                            t['cx'], t['cy'], depth_m, intrinsic)

            active = [t for t in tracks if t['lost'] == 0 and t['z3d'] > 0]
            nearest = min(active, key=lambda t: t['z3d']) if active else None

            # 更新共享目标
            with lock:
                if nearest:
                    nearest_cam[0] = [nearest['x3d'], nearest['y3d'], nearest['z3d']]
                else:
                    nearest_cam[0] = None

            # 显示
            display = color_image.copy()
            draw_results(display, tracks, nearest)

            with lock:
                following = g_pressed[0]

            status = "FOLLOWING" if following else "HOLD (press g to follow)"
            color  = (0, 255, 0) if following else (0, 200, 255)
            cv2.putText(display, status, (10, HEIGHT - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if nearest:
                p_cam = np.array([nearest['x3d'], nearest['y3d'], nearest['z3d']])
                cv2.putText(display,
                            f"cam: [{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]",
                            (10, HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Red Frisbee Tracking', np.hstack((display, mask_bgr)))
            else:
                cv2.imshow('Red Frisbee Tracking', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask

    except KeyboardInterrupt:
        print("\n\n程序被中断")
    finally:
        stop_ctrl.set()
        ctrl_thread.join(timeout=1.0)
        kb_listener.stop()
        cv2.destroyAllWindows()
        pipeline.stop()
        print("\n返回零位...")
        robot.Joint_Pos_Vel([0.0]*robot.motor_count, JOINT_VEL, MAX_TORQUE, iswait=True)
        robot.Joint_Pos_Vel([0.0]*robot.motor_count, [0.0]*robot.motor_count, iswait=False)
        print("程序退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()