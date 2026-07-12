#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panthera-HT 飞盘 hackathon 综合 Demo（总控脚本）

整体流程：
  阶段0  初始化 D405 相机 + Panthera 机械臂，回到 HOME
  阶段1  视觉伺服接近红色飞盘，末端持续跟随飞盘位置移动；操作者可按 f 随时暂停/
         恢复自动跟随（暂停时末端原地悬停，仍有重力补偿），确认位置后按空格键
         手动触发抓取（不再是深度阈值自动触发）；控制台会持续打印飞盘的正面/
         侧面识别信息（跟 4_red_disc_detection.py 的 main() 打印风格一致）
         （复用 4_red_disc_detection.py / 4_red_frisbee_FollowGrasp.py 里的
           CircleTracker / detect_red_disc / get_median_depth / pixel_to_3d /
           draw_results / camera_point_to_base 逻辑）
  阶段2  关节1（索引0，基座旋转）缓慢正转环视一圈，用 YOLO-Pose 判断是否有
         队友举手（复用 red_teammate_detection.py 里的检测判据），发现目标后
         立刻停转、锁定并记录当前姿态
  阶段3  在关节限位保护下，在阶段2锁定的姿态基础上关节1再正转90°，作为这次
         投掷动作的"零位置"
  阶段4  把预先示教好的 arm-only jsonl 主轨迹，在关节1上整体叠加一个常数角度
         偏移（= 当前零位置的实际J1角度 − 轨迹第0帧记录的J1角度），其余关节
         位置和所有关节速度都不变（这只是绕基座转了一个坐标系，速度不受常数
         偏移影响），偏移前先做越界检查，然后按 2_replay.py 的核心回放逻辑
         （倍速 + 重力/摩擦补偿 + 按释放帧控制夹爪张开）执行一次示教动作
  阶段5  回到阶段3建立的"零位置"，结束程序

红色物体识别（阶段1、阶段2）弹窗都支持按 'm' 实时切换掩膜叠加显示，按 'q'
可随时安全退出整个程序。

============================================================================
使用前必须确认/调整的几件事（标了 TODO 的地方）
============================================================================
  1. 下面的 import 假设 4_red_disc_detection.py / red_teammate_detection.py /
     Panthera_lib.py / hand_eye_calibration.json 都跟这个脚本在同一目录下。
     如果你的真实模块文件名不带数字前缀（比如叫 red_disc_detection.py），
     直接用普通 import 也可以，我在下面做了"先尝试普通 import，找不到再按路径
     动态加载"的兼容处理，两种情况都能跑。
  2. 阶段1抓取改为手动按空格触发（不再用深度阈值自动判断"够近了"）；
     SCAN_JOINT1_SPEED（阶段2环视角速度）仍需要你在真机上实测调一版。
  3. TRAJECTORY_FILE / RELEASE_FRAME / SPEED 三个值我直接沿用了你
     2_replay.py 里最新调好的参数，如果你换了新的轨迹文件记得同步改这里。
============================================================================
"""

import os
import sys
import json
import time
import math
import importlib
import importlib.util

import numpy as np
import cv2
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 兼容两种模块命名方式的动态加载：优先普通 import，找不到再按文件路径加载
# ---------------------------------------------------------------------------
def _load_module(module_name, fallback_filename):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        fallback_path = os.path.join(SCRIPT_DIR, fallback_filename)
        if not os.path.isfile(fallback_path):
            raise ImportError(
                f"找不到模块 {module_name}，也找不到备用文件 {fallback_path}，"
                f"请确认对应脚本和本脚本放在同一目录下。"
            )
        spec = importlib.util.spec_from_file_location(module_name, fallback_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module


_disc_mod = _load_module("red_disc_detection", "4_red_disc_detection.py")
_teammate_mod = _load_module("red_teammate_detection", "red_teammate_detection.py")

CircleTracker = _disc_mod.CircleTracker
detect_red_disc = _disc_mod.detect_red_disc
get_median_depth = _disc_mod.get_median_depth
pixel_to_3d = _disc_mod.pixel_to_3d
draw_results = _disc_mod.draw_results

preprocess_red_mask = _teammate_mod.preprocess_red_mask
torso_roi_from_keypoints_or_bbox = _teammate_mod.torso_roi_from_keypoints_or_bbox
red_ratio_in_roi = _teammate_mod.red_ratio_in_roi
hand_raise_diag = _teammate_mod.hand_raise_diag
clip_bbox = _teammate_mod.clip_bbox
TEAMMATE_MODEL_PATH = _teammate_mod.MODEL_PATH
PERSON_CONF_THRES = _teammate_mod.PERSON_CONF_THRES
RED_RATIO_THRES = _teammate_mod.RED_RATIO_THRES

from ultralytics import YOLO
from Panthera_lib import Panthera


# =============================================================================
# 相机配置（跟 4_red_disc_detection.py / 4_red_frisbee_FollowGrasp.py 保持一致）
# =============================================================================
WIDTH, HEIGHT, FPS = 640, 480, 30
MIN_DEPTH_MM, MAX_DEPTH_MM = 70, 1000

# =============================================================================
# 阶段0：HOME 位置 / 通用力矩限幅
# =============================================================================
# 注意：这一步是直接下发关节空间目标角度（Joint_Pos_Vel），不经过正运动学/逆运动学，
# FK 只在阶段1里用来把"到达HOME后的末端姿态"换算成笛卡尔位姿，供IK视觉伺服用。
#
# HOME_JOINT_POS 沿用 4_red_frisbee_FollowGrasp.py 里验证过的姿态——这个姿态和
# 由它算出来的 home_rotation 直接决定了阶段1视觉伺服IK解不解得出来，不能随便改。
# 之前的版本里把"关节1转到最负端限位"当成了阶段0/阶段1的起始姿态，导致阶段1视觉
# 伺服时底座已经转到了-131°附近的极限姿态，跟这个姿态validate过的朝向完全对不上，
# IK自然大概率无解（"DLS逆解迭代…关节角度超出限位"刷屏）。"转到最负端限位"这件事
# 现在只用在阶段2扫描开始前（见 scan_for_teammate 里 get_safe_joint1_bounds 的用法），
# 不再影响阶段0/阶段1。
HOME_JOINT_POS = [0.0, 0.785, 0.785, 0.0, 0.0, 0.0]
# 之前用"限位 + 一个小余量(比如0.10rad)"的方式贴着关节1的真实限位走，结果真机上
# 实际限位跟我们看到的 Follower.yaml 数值对不上（或者别的原因），一小点余量根本
# 不够，导致 Joint_Pos_Vel 直接拒绝执行。与其继续跟"真实限位具体是多少"这个细节
# 死磕，改成一个更省心的思路：任何需要"贴着关节1限位走"的指令，统一只用官方限位
# 的 JOINT_LIMIT_SAFETY_FACTOR 那一部分（比如限位是±2.4rad，0.8的话就只走到
# ±1.92rad），留出足够大的安全垫，不用再逐个margin去凑。
# 之前0.8太保守了（关节1限位±2.4rad时只走到±1.92rad，扫描范围损失太多）。
# 调回更接近实际限位、留一点余量的比例：±2.4rad * 0.958 ≈ ±2.3rad，跟你之前
# 手动用"限位+0.1rad余量"时的效果基本一致，但写法上还是统一走比例系数，
# 如果你的真实限位不是±2.4rad，这个比例仍然会按你实际的限位等比例留出余量。
JOINT_LIMIT_SAFETY_FACTOR = 0.958   # 越小越保守，1.0就是完全贴着真实限位走(不推荐)
# 注意：最大力矩不在这里写死。之前的版本里 MAX_TORQUE 是我猜的一个数值，
# 现在看到 Follower.yaml 才知道 robot.max_torque = [21,36,36,21,10,10]。
# 与其在这里手抄一份容易和yaml失步，不如直接用 robot.max_torque（Panthera
# 在 __init__ 时已经从 Follower.yaml 加载好了），下面凡是需要"默认最大力矩"
# 的地方一律传 max_tqu=None，让机械臂自己用配置文件里的真实值。
JOINT_VEL      = [0.5] * 6      # 阶段5"回到零位置"用的速度
HOME_JOINT_VEL = [0.15] * 6     # 移动到起始位置/扫描起点用的速度，不求快，能到位就行


def get_safe_joint1_bounds(robot):
    """
    返回关节1的"保守版"安全边界：不用完整量程，只用 JOINT_LIMIT_SAFETY_FACTOR
    那一部分，避免每次都卡在真实限位边缘、跟 Panthera 内置的限位保护打架。
    如果限位没加载成功，返回 (None, None)。
    """
    if robot.joint_limits is None:
        return None, None
    lower0 = float(robot.joint_limits['lower'][0]) * JOINT_LIMIT_SAFETY_FACTOR
    upper0 = float(robot.joint_limits['upper'][0]) * JOINT_LIMIT_SAFETY_FACTOR
    return lower0, upper0


def print_joint_target_vs_limits(robot, label, target_pos):
    """
    打印六个关节的目标角度，跟机械臂真实加载的 joint_limits（来自 Follower.yaml）
    逐个对比，每个关节标出 OK / 超限，方便肉眼直接确认，而不用去翻
    Joint_Pos_Vel 内部那段警告打印或者继续猜测具体数值。
    """
    target_pos = [float(v) for v in target_pos]
    print(f"[{label}] 目标关节角度 vs 真实限位:")
    if robot.joint_limits is None:
        print(f"    (未加载关节限位，无法比对) 目标: {np.round(target_pos, 3)}")
        return
    lower = robot.joint_limits['lower']
    upper = robot.joint_limits['upper']
    for i, pos in enumerate(target_pos):
        in_range = lower[i] <= pos <= upper[i]
        flag = "OK" if in_range else "!! 超限 !!"
        print(f"    关节{i + 1}: 目标={pos:.4f}  限位=[{lower[i]:.4f}, {upper[i]:.4f}]  {flag}")


# Joint_Pos_Vel(iswait=True) 最终返回的是 wait_for_position() 的结果，不是"目标位置
# 是否超限"的结果——位置检查通过后指令已经真的发给电机了，wait_for_position() 只是
# 在这个 timeout 时间内反复确认有没有真的走到。默认15秒对"慢速+大角度"的组合不够用，
# 直接给一个足够大的固定值，比每次去算"距离÷速度"更省事。
SLOW_MOVE_TIMEOUT_S = 60.0

# =============================================================================
# 阶段1：视觉伺服抓取参数（沿用 4_red_frisbee_FollowGrasp.py 的取值）
# =============================================================================
X_OFFSET     = -0.15   # 基坐标系下，末端相对飞盘 X 轴向后偏移 15cm
Z_OFFSET     = 0.03    # 基坐标系下，末端相对飞盘 Z 轴向上偏移 3cm
SMOOTH_ALPHA = 0.15    # 目标位置指数平滑系数
GRASP_KP = [12.0, 24.0, 27.0, 12.0, 9.0, 6.0]
GRASP_KD = [2.5,  5.0,  5.5,  2.5,  2.0,  1.3]

# 抓取改为手动触发（按空格），不再靠 z3d 深度阈值自动判断"够近了"
GRASP_SETTLE_TIME_S   = 1.0    # 夹爪闭合后的稳定等待时间
STAGE1_CAMERA_STALL_TIMEOUT_S = 5.0   # 连续取不到有效相机帧超过这个时间就报错退出
STAGE1_NO_GRASP_TIMEOUT_S     = 120.0  # 阶段1总时长超过这个时间还没抓到就报错退出

# =============================================================================
# 阶段2：环视扫描参数
# =============================================================================
SCAN_JOINT1_SPEED  = 0.08   # rad/s，缓慢正转角速度，不求快，TODO按真机实测手感调整
SCAN_LIMIT_MARGIN  = 0.05   # rad，离限位还剩多少就掉头反向继续扫
SCAN_TIMEOUT_S      = 60.0  # 超时仍未发现举手队友就报错退出，避免无限空转
TEAMMATE_CONFIRM_FRAMES = 8   # 连续多少帧都判定为"举手目标"才真正停止，防止单帧误判
STATUS_PRINT_INTERVAL_S = 1.0  # 状态打印节流间隔（秒），避免刷屏
# 阶段2改用跟其余阶段一致的 pos_vel_tqe_kp_kd（MIT模式）做位置渐进，而不是 Joint_Vel，
# 这里的 kp/kd 沿用阶段1抓取时的量级（较柔顺），TODO按真机手感微调
SCAN_KP = [12.0, 24.0, 27.0, 12.0, 9.0, 6.0]
SCAN_KD = [2.5,  5.0,  5.5,  2.5,  2.0,  1.3]

# =============================================================================
# 阶段3：限位保护下正转90°
# =============================================================================
ROTATE90_VEL = [0.3] * 6
# 力矩同样不写死，Joint_Pos_Vel 传 None 时会自动用 robot.max_torque（来自 Follower.yaml）

# =============================================================================
# 阶段4：轨迹回放参数（沿用 2_replay.py 里最新调好的取值）
# =============================================================================
TRAJECTORY_FILE = os.path.join(SCRIPT_DIR, "arm_teach_20260711_005433.jsonl")
RELEASE_FRAME   = 18293
SPEED           = 2.6

REPLAY_KP = [30.0, 40.0, 55.0, 15.0, 7.0, 5.0]
REPLAY_KD = [3.0, 4.0, 5.5, 1.5, 0.7, 0.5]
REPLAY_START_JOINT_VEL = [0.5] * 6
# 注意：这个不是 robot.max_torque，是 2_replay.py 里原本就故意调得比
# robot.max_torque([21,36,36,21,10,10]) 更保守的一组回放专用力矩上限，
# 只用于夹爪/摩擦补偿力矩的clip，照抄原脚本数值，不是没写清楚的占位值。
REPLAY_TAU_LIMIT       = [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]

FC = [0.15, 0.12, 0.12, 0.12, 0.04, 0.04]
FV = [0.05, 0.05, 0.05, 0.03, 0.02, 0.02]
VEL_THRESHOLD = 0.02

GRIPPER_HOLD_POS    = 0.0
GRIPPER_RELEASE_POS = 1.5
GRIPPER_VEL    = 0.0
GRIPPER_TORQUE = 0.0
GRIPPER_KP = 70
GRIPPER_KD = 0.2


# =============================================================================
# 相机初始化（跟已有脚本一致）
# =============================================================================
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

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


def camera_point_to_base(point_camera, robot, T_tcp_camera):
    """相机坐标系 -> 基坐标系（跟 4_red_frisbee_FollowGrasp.py 完全一致）"""
    fk = robot.forward_kinematics()
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = np.array(fk['rotation'])
    T_base_tcp[:3, 3] = np.array(fk['position'])
    T_base_camera = T_base_tcp @ T_tcp_camera
    p_homo = np.append(np.array(point_camera), 1.0)
    return (T_base_camera @ p_homo)[:3]


# =============================================================================
# 阶段1：视觉伺服接近并抓取红色飞盘
# =============================================================================
def approach_and_grasp(robot, pipeline, align, intrinsic, depth_scale,
                        min_depth_units, max_depth_units, T_tcp_camera, home_rotation):
    print("\n[阶段1] 张开夹爪，准备抓取...")
    robot.gripper_open(vel=3.0)
    time.sleep(GRASP_SETTLE_TIME_S)

    print("[阶段1] 开始视觉伺服接近红色飞盘...（按 f 停止/恢复跟随，按空格键抓取，"
          "按 q 退出程序，按 m 切换掩膜）")

    window_name = "Stage1 - Detect & Grasp Frisbee"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)
    show_mask = False
    following = True   # 按 f 可以暂停/恢复自动视觉伺服跟随，暂停时末端原地悬停(仍有重力补偿)

    tracker = CircleTracker()
    last_joint = list(robot.get_current_pos())
    smoothed_target = None
    last_status_print = 0.0
    stage_start_time = time.time()
    last_frame_ok_time = time.time()

    try:
        while True:
            # ── 无论这一帧相机数据是否有效，都先处理按键 ──
            # 之前的版本里，取帧失败会直接 continue，导致 cv2.waitKey() 根本执行不到，
            # 相机一旦抖动/掉线就会变成真正意义上按键都响应不了的死循环。现在把按键检测
            # 挪到取帧校验之前，保证任何时候都能响应 f/空格/'q'/'m'。
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt("用户在阶段1请求退出")
            elif key == ord('m'):
                show_mask = not show_mask
            elif key == ord('f'):
                # ── 按 f 停止/恢复自动跟随 ──
                # 停止后不再更新 smoothed_target/重新解IK，末端会原地悬停(仍持续下发
                # 重力补偿，不会掉下来)；恢复后从当前实际位置重新开始跟随，不会因为
                # 暂停期间飞盘挪动了位置而"猛地"跳过去。
                following = not following
                if following:
                    print("[阶段1] 恢复自动跟随")
                    smoothed_target = None   # 重新从当前目标开始平滑，避免恢复瞬间跳变
                else:
                    print("[阶段1] 已停止自动跟随，末端原地悬停。可手动确认位置后按空格抓取，"
                          "或再按一次 f 恢复跟随")
            elif key == ord(' '):
                # ── 抓取改为手动触发：不再靠 z3d 自动判断"够近了"，由操作者肉眼判断按空格 ──
                print("[阶段1] 检测到空格键，手动触发抓取")
                robot.gripper_close(vel=3.0)
                time.sleep(GRASP_SETTLE_TIME_S)
                print("[阶段1] 抓取完成")
                return

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                if time.time() - last_frame_ok_time > STAGE1_CAMERA_STALL_TIMEOUT_S:
                    raise RuntimeError(
                        f"阶段1: 相机连续 {STAGE1_CAMERA_STALL_TIMEOUT_S:.0f}s 未取到有效帧，"
                        f"请检查 RealSense 连接（USB passthrough / usbipd 等）"
                    )
                continue
            last_frame_ok_time = time.time()

            if time.time() - stage_start_time > STAGE1_NO_GRASP_TIMEOUT_S:
                raise RuntimeError(
                    f"阶段1: 超过 {STAGE1_NO_GRASP_TIMEOUT_S:.0f}s 仍未按空格完成抓取，"
                    f"请检查飞盘是否在视野内，或直接按空格手动抓取"
                )

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

            # ── 显示 ──
            display = color_image.copy()
            draw_results(display, tracks, nearest)
            status_text = "FOLLOWING" if following else "PAUSED (press f to resume)"
            status_color = (0, 255, 0) if following else (0, 200, 255)
            cv2.putText(display, f"Stage1: {status_text}",
                        (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, "f=pause/resume  SPACE=grasp  q=quit  m=mask",
                        (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            if show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow(window_name, np.hstack((display, mask_bgr)))
            else:
                cv2.imshow(window_name, display)

            # ── 无论是否在跟随，都持续下发重力补偿，保持末端稳定悬停 ──
            robot_gra = robot.get_Gravity()
            robot_torque = np.array(robot_gra)

            if not following:
                # 暂停跟随：只维持当前 last_joint 悬停，不更新目标、不重新解IK
                robot.pos_vel_tqe_kp_kd(last_joint, [0.0] * robot.motor_count,
                                         robot_torque, GRASP_KP, GRASP_KD)
                continue

            if nearest is None:
                if time.time() - last_status_print > STATUS_PRINT_INTERVAL_S:
                    print("[阶段1] 状态: 未检测到红色飞盘")
                    last_status_print = time.time()
                continue

            # ── 视觉伺服：把末端驱动到飞盘附近的抓取位姿 ──
            p_cam = np.array([nearest['x3d'], nearest['y3d'], nearest['z3d']])
            p_base = camera_point_to_base(p_cam, robot, T_tcp_camera)

            target_pos = p_base.copy()
            target_pos[0] += X_OFFSET
            target_pos[2] += Z_OFFSET

            if smoothed_target is None:
                smoothed_target = target_pos.copy()
            else:
                smoothed_target = SMOOTH_ALPHA * target_pos + (1 - SMOOTH_ALPHA) * smoothed_target

            joint_pos = robot.inverse_kinematics(
                smoothed_target.tolist(), home_rotation, last_joint, multi_init=False
            )
            if joint_pos is not None:
                last_joint = joint_pos

            robot.pos_vel_tqe_kp_kd(last_joint, [0.0] * robot.motor_count,
                                     robot_torque, GRASP_KP, GRASP_KD)

            # ── 状态打印：跟 4_red_disc_detection.py 的 main() 一样，区分正面/侧面打印 ──
            if time.time() - last_status_print > STATUS_PRINT_INTERVAL_S:
                if nearest['view'] == 'side':
                    print(f"[阶段1] 状态: [侧面] 中心({nearest['cx']},{nearest['cy']}) "
                          f"角度={nearest['angle_deg']:.1f}° z3d={nearest['z3d']:.4f}m "
                          f"(按空格键抓取)")
                else:
                    print(f"[阶段1] 状态: [正面] 中心({nearest['cx']},{nearest['cy']}) "
                          f"z3d={nearest['z3d']:.4f}m (按空格键抓取)")
                last_status_print = time.time()

    finally:
        cv2.destroyWindow(window_name)


# =============================================================================
# 阶段2：关节1缓慢正转环视，YOLO-Pose 识别举手队友
# =============================================================================
def scan_for_teammate(robot, pipeline, align, yolo_model):
    print("\n[阶段2] 关节1缓慢正转环视，寻找举手队友...（按 q 退出程序，按 m 切换掩膜）")

    window_name = "Stage2 - Scan for Raised-Hand Teammate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)
    show_mask = False

    lower0, upper0 = get_safe_joint1_bounds(robot)   # 只用 JOINT_LIMIT_SAFETY_FACTOR 那一部分量程
    if robot.joint_limits is not None:
        raw_lower0 = float(robot.joint_limits['lower'][0])
        raw_upper0 = float(robot.joint_limits['upper'][0])
        print(f"[阶段2] 关节1真实限位: [{raw_lower0:.3f}, {raw_upper0:.3f}] rad，"
              f"缩放({JOINT_LIMIT_SAFETY_FACTOR}倍)后安全边界: [{lower0:.3f}, {upper0:.3f}] rad")
    # ── 阶段1→阶段2 过渡：分两步走，不能直接把关节1转到极限位置 ──
    # 阶段1视觉伺服抓取时，关节2~6会跟着IK解出来停在各种角度上；如果直接在这个
    # "形状不定"的姿态基础上把关节1转到最负端限位，可能因为整个臂的姿态不同，
    # 导致联动到其他关节超限（或者更容易在真实空间里撞限位/自碰撞）。所以先让
    # 关节2~6独立回到零位（关节1先保持不动），姿态确定之后，再单独转动关节1到
    # 扫描起始位置——两步都是慢速(HOME_JOINT_VEL)、不求快只求到位。
    post_grasp_pos = list(robot.get_current_pos())

    zero_rest_pos = [post_grasp_pos[0]] + [0.0] * 5
    print(f"[阶段2] 第一步: 关节2~6先回零位(关节1暂不动): {np.round(zero_rest_pos, 3)}")
    print_joint_target_vs_limits(robot, "阶段2-第一步", zero_rest_pos)
    ok = robot.Joint_Pos_Vel(zero_rest_pos, HOME_JOINT_VEL, None, iswait=True,
                             timeout=SLOW_MOVE_TIMEOUT_S)
    if not ok:
        raise RuntimeError(
            "阶段2: 关节2~6回零位失败——六个关节目标位置本身没有超限（见上面打印），"
            "大概率是 wait_for_position() 在算出来的timeout内没等到实际到位"
            "（比如物理卡住/夹着的飞盘干涉/速度设置太慢配合关节角度大导致的），"
            "不是限位保护拒绝，请检查机械臂实际有没有真的在动"
        )

    other_pos = [0.0] * 5   # 扫描全程关节2~6保持在零位

    if lower0 is not None:
        scan_start_j1 = lower0
    else:
        scan_start_j1 = zero_rest_pos[0]
        print("[阶段2] 警告: 未加载关节限位，扫描起始角度退化为当前角度，请自行确认安全")

    scan_start_pos = [scan_start_j1] + other_pos
    print(f"[阶段2] 第二步: 关节2~6已回零，转动关节1到扫描起始位置(={scan_start_j1:.3f}rad，"
          f"关节1限位的{JOINT_LIMIT_SAFETY_FACTOR*100:.0f}%处): {np.round(scan_start_pos, 3)}")
    print_joint_target_vs_limits(robot, "阶段2-第二步", scan_start_pos)
    ok = robot.Joint_Pos_Vel(scan_start_pos, HOME_JOINT_VEL, None, iswait=True,
                             timeout=SLOW_MOVE_TIMEOUT_S)
    if not ok:
        raise RuntimeError(
            "阶段2: 移动到扫描起始位置失败——六个关节目标位置本身没有超限（见上面打印），"
            "大概率是 wait_for_position() 在算出来的timeout内没等到实际到位"
            "（比如转动距离较大、速度设置较慢导致实际耗时超出预期，或者机械臂物理上"
            "卡住了），不是限位保护拒绝。请先确认机械臂运行过程中是否真的在转动"
        )

    # ── 注意：这里不用 Joint_Vel() ──
    # 你现有的所有脚本（记录/回放/抓取）全程只用 pos_vel_tqe_kp_kd（MIT五参数模式）
    # 控制机械臂。阶段1结束时机械臂正是被 pos_vel_tqe_kp_kd 用较高的kp/kd锁在抓取
    # 位置的（MIT位置闭环）。Joint_Vel() 内部走的是另一条底层指令路径
    # （motor.velocity()），跟 MIT 模式不是同一套闭环，电机很可能仍停留在上一条
    # MIT指令的位置保持状态，导致 Joint_Vel() 发的纯速度指令根本没有生效——这正是
    # "阶段2完全不动"的原因。改成跟其余阶段完全一致的 pos_vel_tqe_kp_kd：让关节1的
    # 目标位置按（用实际测得的循环间隔dt累积的）小步长持续递增，效果上等价于缓慢
    # 匀速转动，但走的是已经在你机械臂上验证能work的控制路径。
    scan_target_j1 = scan_start_j1

    direction = 1
    scan_start_time = time.time()
    last_loop_time = time.time()
    last_status_print = 0.0
    confirm_count = 0

    try:
        while True:
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now

            if now - scan_start_time > SCAN_TIMEOUT_S:
                hold_pos = [scan_target_j1] + other_pos
                robot.Joint_Pos_Vel(hold_pos, [0.0] * robot.motor_count, iswait=False)
                raise RuntimeError("阶段2: 环视扫描超时仍未发现举手队友，请检查场景或放宽判定阈值")

            # ── 掉头判断：基于"目标位置"而不是反馈位置，避免因控制滞后来回抖动 ──
            if upper0 is not None and direction > 0 and scan_target_j1 >= upper0 - SCAN_LIMIT_MARGIN:
                direction = -1
                print("[阶段2] 关节1接近正向限位，掉头反向继续扫描")
            elif lower0 is not None and direction < 0 and scan_target_j1 <= lower0 + SCAN_LIMIT_MARGIN:
                direction = 1
                print("[阶段2] 关节1接近负向限位，掉头反向继续扫描")

            scan_target_j1 += SCAN_JOINT1_SPEED * direction * dt
            if upper0 is not None:
                scan_target_j1 = min(scan_target_j1, upper0 - SCAN_LIMIT_MARGIN)
            if lower0 is not None:
                scan_target_j1 = max(scan_target_j1, lower0 + SCAN_LIMIT_MARGIN)

            target_pos = [scan_target_j1] + other_pos
            target_vel = [SCAN_JOINT1_SPEED * direction] + [0.0] * 5

            gravity = np.array(robot.get_Gravity(), dtype=float)
            robot.pos_vel_tqe_kp_kd(target_pos, target_vel, gravity.tolist(), SCAN_KP, SCAN_KD)

            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                # 同阶段1的修复：即使这一帧取不到，也要先处理按键，否则相机一旦
                # 抖动/掉线，'q'键会被 continue 挡住，变成真正意义上退不出的死循环。
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    hold_pos = [scan_target_j1] + other_pos
                    robot.Joint_Pos_Vel(hold_pos, [0.0] * robot.motor_count, iswait=False)
                    raise KeyboardInterrupt("用户在阶段2请求退出")
                elif key == ord('m'):
                    show_mask = not show_mask
                continue

            color_image = np.asanyarray(color_frame.get_data())

            result = yolo_model.predict(color_image, conf=PERSON_CONF_THRES, verbose=False)[0]
            red_mask = preprocess_red_mask(color_image)
            display = color_image.copy()

            boxes = result.boxes
            keypoints = result.keypoints
            target_found_this_frame = False
            n_persons = 0 if boxes is None else len(boxes)
            n_red_pass = 0
            n_hand_raised = 0

            if boxes is not None and keypoints is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                kpts_all = keypoints.data.cpu().numpy()

                for i, (box, conf) in enumerate(zip(xyxy, confs)):
                    bbox = clip_bbox(*box)
                    kpts = kpts_all[i] if i < len(kpts_all) else None

                    torso_roi, _ = torso_roi_from_keypoints_or_bbox(kpts, bbox)
                    red_ratio = red_ratio_in_roi(red_mask, torso_roi)
                    red_pass = red_ratio >= RED_RATIO_THRES
                    hand_raised, _ = hand_raise_diag(kpts)

                    if red_pass:
                        n_red_pass += 1
                    if hand_raised:
                        n_hand_raised += 1

                    is_target = (float(conf) >= PERSON_CONF_THRES and red_pass and hand_raised)
                    if is_target:
                        target_found_this_frame = True

                    x1, y1, x2, y2 = bbox
                    box_color = (0, 255, 0) if is_target else (0, 0, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

            # ── 防抖确认计数 ──
            if target_found_this_frame:
                confirm_count += 1
            else:
                confirm_count = 0

            cv2.putText(display, f"Stage2: scanning ({confirm_count}/{TEAMMATE_CONFIRM_FRAMES})",
                        (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # ── 节流状态打印，让操作者随时了解当前状态 ──
            if time.time() - last_status_print > STATUS_PRINT_INTERVAL_S:
                if n_persons == 0:
                    print(f"[阶段2] 状态: 画面中未检测到人 (关节1目标角度={scan_target_j1:.3f}rad)")
                else:
                    print(f"[阶段2] 状态: 检测到 {n_persons} 个候选人体框，"
                          f"红衣达标 {n_red_pass} 个，举手 {n_hand_raised} 个，"
                          f"确认帧 {confirm_count}/{TEAMMATE_CONFIRM_FRAMES} "
                          f"(关节1目标角度={scan_target_j1:.3f}rad)")
                last_status_print = time.time()

            if show_mask:
                mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow(window_name, np.hstack((display, mask_bgr)))
            else:
                cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                hold_pos = [scan_target_j1] + other_pos
                robot.Joint_Pos_Vel(hold_pos, [0.0] * robot.motor_count, iswait=False)
                raise KeyboardInterrupt("用户在阶段2请求退出")
            elif key == ord('m'):
                show_mask = not show_mask

            if confirm_count >= TEAMMATE_CONFIRM_FRAMES:
                current_pos = [scan_target_j1] + other_pos
                print(f"[阶段2] 连续 {TEAMMATE_CONFIRM_FRAMES} 帧确认举手队友，停止扫描。"
                      f"当前关节角度: {np.round(current_pos, 3)}")
                # 用位置控制锁死在这一刻，而不是继续零刚度自由漂移
                robot.Joint_Pos_Vel(current_pos, [0.0] * robot.motor_count, iswait=True)
                return current_pos

    finally:
        cv2.destroyWindow(window_name)


# =============================================================================
# 阶段3：限位保护下，在当前姿态基础上关节1正转90°
# =============================================================================
def rotate_90_with_limit_check(robot, current_pos):
    target_pos = list(current_pos)
    desired_j1 = current_pos[0] + math.pi / 2.0

    lower0, upper0 = get_safe_joint1_bounds(robot)   # 只用 JOINT_LIMIT_SAFETY_FACTOR 那一部分量程
    if lower0 is not None:
        clipped_j1 = min(max(desired_j1, lower0), upper0)
        if abs(clipped_j1 - desired_j1) > 1e-9:
            print(f"[阶段3] 警告: 正转90°会超出关节1的保守边界 [{lower0:.3f}, {upper0:.3f}] rad "
                  f"(官方限位的{JOINT_LIMIT_SAFETY_FACTOR*100:.0f}%)，"
                  f"已从 {desired_j1:.3f} 裁剪为 {clipped_j1:.3f} rad")
        target_pos[0] = clipped_j1
    else:
        print("[阶段3] 警告: 未加载关节限位信息，无法自动裁剪，请自行确认90°转动安全")
        target_pos[0] = desired_j1

    print(f"[阶段3] 关节1正转: {current_pos[0]:.3f} -> {target_pos[0]:.3f} rad")
    print_joint_target_vs_limits(robot, "阶段3", target_pos)
    ok = robot.Joint_Pos_Vel(target_pos, ROTATE90_VEL, None, iswait=True,
                             timeout=SLOW_MOVE_TIMEOUT_S)
    if not ok:
        raise RuntimeError(
            "阶段3: 转动90°失败——目标位置本身没有超限（见上面打印），大概率是"
            "wait_for_position() 超时没等到实际到位，不是限位保护拒绝，"
            "请检查机械臂运行过程中是否真的在转动"
        )

    zero_pos = list(robot.get_current_pos())
    print(f"[阶段3] 已到达新的'零位置': {np.round(zero_pos, 3)}")
    return zero_pos


# =============================================================================
# 阶段4：轨迹关节1整体偏移 + 回放（核心逻辑照搬 2_replay.py）
# =============================================================================
def load_frames(filepath):
    frames = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "metadata":
                continue
            if "pos" not in item and "positions" in item:
                item["pos"] = item["positions"]
            if "vel" not in item and "velocities" in item:
                item["vel"] = item["velocities"]
            if "pos" in item and "t" in item:
                frames.append(item)
    if not frames:
        raise ValueError("轨迹文件里没有有效帧：需要包含 t 和 pos/positions")
    return frames


def friction_compensation(vel):
    tau = []
    for v, fc, fv in zip(vel, FC, FV):
        if abs(v) < VEL_THRESHOLD:
            tau.append(0.0)
        else:
            tau.append(fc * np.sign(v) + fv * v)
    return np.array(tau, dtype=float)


def limit_torque(tau):
    tau = np.array(tau, dtype=float)
    limit = np.array(REPLAY_TAU_LIMIT, dtype=float)
    return np.clip(tau, -limit, limit)


def apply_joint1_offset(frames, delta_j1):
    """
    整条轨迹只在关节1(索引0)的位置上叠加一个常数偏移 delta_j1：
    从"原始录制起始姿态"到"当前新零位置"之间只发生了一次绕基座的旋转，
    其余关节的相对位形不变；由于偏移是常数，其对时间的导数为0，所以所有
    关节记录的角速度都不需要改变——这正是"坐标系旋转"这件事在关节空间里
    最简单的体现。
    """
    offset_frames = []
    for fr in frames:
        new_fr = dict(fr)
        new_pos = list(fr["pos"])
        new_pos[0] = new_pos[0] + delta_j1
        new_fr["pos"] = new_pos
        offset_frames.append(new_fr)
    return offset_frames


def check_trajectory_within_limits(robot, frames):
    if robot.joint_limits is None:
        print("[阶段4] 警告: 未加载关节限位，跳过轨迹越界检查")
        return True

    lower = np.asarray(robot.joint_limits['lower'])
    upper = np.asarray(robot.joint_limits['upper'])
    all_pos = np.array([fr["pos"] for fr in frames])
    out_of_range = np.logical_or(all_pos < lower, all_pos > upper)

    if np.any(out_of_range):
        bad_joint_idx = np.where(np.any(out_of_range, axis=0))[0]
        print("[阶段4] 错误: 偏移后的轨迹在以下关节上超出限位，已中止回放：")
        for j in bad_joint_idx:
            vals = all_pos[:, j]
            print(f"  关节{j + 1}: 轨迹范围[{vals.min():.3f}, {vals.max():.3f}] "
                  f"超出限位[{lower[j]:.3f}, {upper[j]:.3f}]")
        return False
    return True


def replay_with_offset(robot, frames, speed, release_frame):
    """
    核心回放逻辑照搬 2_replay.py 的 replay_original_speed()，一个字都没有因为
    "扫描阶段要慢"而改动：倍速(speed)缩放的是录制时间戳和角速度，不是新加的
    限速逻辑；这里的轨迹只是在阶段4之前被 apply_joint1_offset() 做了一次关节1
    常数偏移（坐标系旋转），除此之外时间轴、速度、KP/KD、摩擦/重力补偿、
    释放帧逻辑全部和 2_replay.py 一致。HOME_JOINT_VEL / SCAN_JOINT1_SPEED 这些
    "慢速"常量只用于阶段0和阶段2，不会传到这个函数里。
    """
    print(f"[阶段4] 共 {len(frames)} 帧，回放倍速: {speed}x，释放帧: {release_frame}")
    print("[阶段4] 正在移动到(偏移后的)轨迹起点...")

    start_pos = frames[0]["pos"]
    robot.Joint_Pos_Vel(start_pos, REPLAY_START_JOINT_VEL, REPLAY_TAU_LIMIT, iswait=True)
    time.sleep(0.5)

    print("[阶段4] 开始回放示教动作...")
    t0_record = frames[0]["t"]
    t0_wall = time.time()
    prev_record_t = t0_record
    cumulative_target_t = 0.0

    for i, frame in enumerate(frames):
        dt_raw = frame["t"] - prev_record_t
        dt_scaled = dt_raw / speed if speed > 0 else dt_raw
        cumulative_target_t += dt_scaled
        prev_record_t = frame["t"]

        now_t = time.time() - t0_wall
        wait_t = cumulative_target_t - now_t
        if wait_t > 0:
            time.sleep(wait_t)

        pos = frame["pos"]
        vel_raw = frame.get("vel", [0.0] * robot.motor_count)
        vel = [v * speed for v in vel_raw]

        gravity = np.array(robot.get_Gravity(), dtype=float)
        friction = friction_compensation(vel)
        torque = limit_torque(gravity + friction)

        robot.pos_vel_tqe_kp_kd(pos, vel, torque.tolist(), REPLAY_KP, REPLAY_KD)

        if i < release_frame:
            robot.gripper_control_MIT(GRIPPER_HOLD_POS, GRIPPER_VEL, GRIPPER_TORQUE,
                                       GRIPPER_KP, GRIPPER_KD)
        else:
            robot.gripper_control_MIT(GRIPPER_RELEASE_POS, GRIPPER_VEL, GRIPPER_TORQUE,
                                       GRIPPER_KP, GRIPPER_KD)

        if i % 500 == 0:
            print(f"[阶段4] frame {i}/{len(frames) - 1}")

    print("[阶段4] 回放完成")


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 70)
    print("Panthera-HT 飞盘接抛 hackathon 综合 Demo")
    print("=" * 70)

    if not os.path.isfile(TRAJECTORY_FILE):
        print(f"[阶段0] 错误: 找不到轨迹文件 {TRAJECTORY_FILE}")
        sys.exit(1)

    calib_path = os.path.join(SCRIPT_DIR, "hand_eye_calibration.json")
    if not os.path.isfile(calib_path):
        print(f"[阶段0] 错误: 找不到手眼标定文件 {calib_path}")
        sys.exit(1)
    with open(calib_path, "r") as f:
        calib = json.load(f)
    T_tcp_camera = np.array(calib["T_tcp_camera"])

    print("\n[阶段0] 初始化 RealSense D405 相机...")
    pipeline, align, intrinsic, depth_scale, min_depth_units, max_depth_units = init_realsense()

    print("\n[阶段0] 初始化机械臂...")
    config_path = os.path.join(SCRIPT_DIR, "../robot_param/Follower.yaml")
    robot = Panthera(config_path)

    # ── 用真实加载的 Follower.yaml 数值核对本脚本里手写的几个速度常量 ──
    # （之前的版本里这几个速度只是凭感觉写的猜测值；现在 robot.velocity_limits
    #  已经从 Follower.yaml 里读出来了，这里做一次显式核对，而不是继续盲猜。）
    if robot.velocity_limits is not None:
        vlim = np.asarray(robot.velocity_limits)
        for name, vel in [("HOME_JOINT_VEL", HOME_JOINT_VEL),
                           ("JOINT_VEL", JOINT_VEL),
                           ("ROTATE90_VEL", ROTATE90_VEL),
                           ("SCAN_JOINT1(索引0)", [SCAN_JOINT1_SPEED] + [0.0] * 5)]:
            v = np.asarray(vel)
            if np.any(np.abs(v) > vlim):
                print(f"[阶段0] 警告: {name}={v.tolist()} 超出 Follower.yaml 里的 "
                      f"velocity_limits={vlim.tolist()}，请调低")
            else:
                print(f"[阶段0] 核对通过: {name}={v.tolist()} 在 velocity_limits="
                      f"{vlim.tolist()} 范围内")
    else:
        print("[阶段0] 警告: 未加载 velocity_limits，无法核对本脚本速度常量是否安全")

    # 阶段0直接用 4_red_frisbee_FollowGrasp.py 验证过的固定HOME姿态，不再动态改关节1
    # ——这个姿态和由它算出来的 home_rotation 直接决定阶段1视觉伺服IK解不解得出来。
    print(f"[阶段0] 移动到初始位置(速度放慢，不求快只求到位): {HOME_JOINT_POS}")
    print_joint_target_vs_limits(robot, "阶段0", HOME_JOINT_POS)
    ok = robot.Joint_Pos_Vel(HOME_JOINT_POS, HOME_JOINT_VEL, None, iswait=True,
                             timeout=SLOW_MOVE_TIMEOUT_S)
    if not ok:
        raise RuntimeError(
            "阶段0: 移动到 HOME_JOINT_POS 失败——目标位置本身没有超限（见上面打印），"
            "大概率是 wait_for_position() 超时没等到实际到位，不是限位保护拒绝，"
            "请检查机械臂运行过程中是否真的在转动"
        )
    time.sleep(0.5)

    fk_home = robot.forward_kinematics()
    home_rotation = np.array(fk_home['rotation'])

    zero_pos = None

    try:
        # ── 阶段1 ──
        approach_and_grasp(robot, pipeline, align, intrinsic, depth_scale,
                            min_depth_units, max_depth_units, T_tcp_camera, home_rotation)

        # ── 阶段2 ──
        print("\n[阶段2] 加载 YOLO-Pose 模型...")
        yolo_model = YOLO(TEAMMATE_MODEL_PATH)
        scan_stop_pos = scan_for_teammate(robot, pipeline, align, yolo_model)

        # ── 阶段3 ──
        zero_pos = rotate_90_with_limit_check(robot, scan_stop_pos)

        # ── 阶段4 ──
        print(f"\n[阶段4] 加载主示教轨迹: {TRAJECTORY_FILE}")
        frames = load_frames(TRAJECTORY_FILE)
        recorded_j1_start = frames[0]["pos"][0]
        delta_j1 = zero_pos[0] - recorded_j1_start
        print(f"[阶段4] 关节1偏移量 delta = {delta_j1:.4f} rad "
              f"(新零位置 {zero_pos[0]:.4f} - 录制起点 {recorded_j1_start:.4f})")

        offset_frames = apply_joint1_offset(frames, delta_j1)

        if not check_trajectory_within_limits(robot, offset_frames):
            raise RuntimeError("阶段4: 偏移后的轨迹超出关节限位，已中止，"
                               "请检查零位置或更换轨迹文件")

        replay_with_offset(robot, offset_frames, SPEED, RELEASE_FRAME)

        # ── 阶段5 ──
        print("\n[阶段5] 回到零位置...")
        print_joint_target_vs_limits(robot, "阶段5", zero_pos)
        ok5 = robot.Joint_Pos_Vel(zero_pos, JOINT_VEL, None, iswait=True,
                                  timeout=SLOW_MOVE_TIMEOUT_S)
        if not ok5:
            print("[阶段5] 警告: 回零位似乎没在预计时间内到位（不是限位问题），"
                  "请检查机械臂当前实际状态")
        print("[阶段5] 完成，demo结束")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        if zero_pos is not None:
            print("\n[收尾] 尝试保持在最后的安全位置...")
            try:
                robot.Joint_Pos_Vel(zero_pos, [0.0] * robot.motor_count, iswait=False)
            except Exception:
                pass
        print("相机与窗口已释放，程序结束")


if __name__ == "__main__":
    main()
