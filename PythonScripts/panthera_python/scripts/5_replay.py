#!/usr/bin/env python3
"""
原速回放 arm-only jsonl 轨迹。
不交互，不选择文件，直接回放 TRAJECTORY_FILE。
"""

import os
import sys
import json
import time
import numpy as np

from Panthera_lib import Panthera


# ---------------- 参数区 ----------------
TRAJECTORY_FILE = "arm_teach_20260711_005433.jsonl"
RELEASE_FRAME = 18293

# 回放倍速：>1 加速，<1 减速，1.0 为原速
SPEED = 2.6


# ---- 帧区间加速 ----
# 只在 [ACCEL_START_FRAME, ACCEL_END_FRAME] 这个帧数段内，
# 在 SPEED 的基础上再乘以 ACCEL_FACTOR 进行加速；区间外仍按 SPEED 正常回放。
ACCEL_START_FRAME = 0
ACCEL_END_FRAME = 0
ACCEL_FACTOR = 1.0

GRIPPER_HOLD_POS = 0.0
GRIPPER_RELEASE_POS = 1.5

GRIPPER_VEL = 0.0
GRIPPER_TORQUE = 0.0
GRIPPER_KP = 70
GRIPPER_KD = 0.2


# 关节 PD 增益
KP = [30.0, 40.0, 55.0, 15.0, 7.0, 5.0]
KD = [3.0, 4.0, 5.5, 1.5, 0.7, 0.5]

# 移动到轨迹起点时用的速度
START_JOINT_VEL = [0.5] * 6

# 力矩限制
TAU_LIMIT = [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]

# 摩擦补偿参数
FC = [0.15, 0.12, 0.12, 0.12, 0.04, 0.04]
FV = [0.05, 0.05, 0.05, 0.03, 0.02, 0.02]
VEL_THRESHOLD = 0.02
# ---------------------------------------


def load_frames(filepath):
    frames = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # 跳过 metadata
            if item.get("type") == "metadata":
                continue

            # 字段兼容：新版 positions/velocities -> 旧版 pos/vel
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
    limit = np.array(TAU_LIMIT, dtype=float)
    return np.clip(tau, -limit, limit)


def get_effective_speed(frame_idx, base_speed):
    """
    返回该帧实际使用的回放速度：
    - 若 frame_idx 落在 [ACCEL_START_FRAME, ACCEL_END_FRAME] 区间内，
      在 base_speed 基础上再乘以 ACCEL_FACTOR
    - 否则按 base_speed 正常回放
    """
    if ACCEL_START_FRAME <= frame_idx <= ACCEL_END_FRAME:
        return base_speed * ACCEL_FACTOR
    return base_speed


def command_gripper(robot, pos):
    robot.gripper_control_MIT(
        pos, GRIPPER_VEL, GRIPPER_TORQUE, GRIPPER_KP, GRIPPER_KD
    )


def replay_original_speed(robot, frames, speed=1.0):
    print(f"[Player] 共 {len(frames)} 帧，回放倍速: {speed}x")
    print(f"[Player] 加速区间: 第 {ACCEL_START_FRAME} 帧 - 第 {ACCEL_END_FRAME} 帧，"
          f"区间内额外加速 {ACCEL_FACTOR}x（等效 {speed * ACCEL_FACTOR}x）")
    print("[Player] 正在移动到轨迹起点...")

    start_pos = frames[0]["pos"]
    robot.Joint_Pos_Vel(start_pos, START_JOINT_VEL, TAU_LIMIT, iswait=True)
    time.sleep(0.5)

    print("[Player] 开始回放...")

    t0_record = frames[0]["t"]
    t0_wall = time.time()

    prev_record_t = t0_record
    cumulative_target_t = 0.0
    in_accel_zone_prev = False

    for i, frame in enumerate(frames):
        eff_speed = get_effective_speed(i, speed)

        # 是否处于加速区间，仅用于打印提示
        in_accel_zone = ACCEL_START_FRAME <= i <= ACCEL_END_FRAME
        if in_accel_zone != in_accel_zone_prev:
            state_str = "进入" if in_accel_zone else "退出"
            print(f"[Player] frame {i}: {state_str}加速区间 (有效倍速 {eff_speed}x)")
            in_accel_zone_prev = in_accel_zone

        # 按局部有效速度累积目标时间，而不是用固定 speed 整体压缩时间戳
        dt_raw = frame["t"] - prev_record_t
        dt_scaled = dt_raw / eff_speed if eff_speed > 0 else dt_raw
        cumulative_target_t += dt_scaled
        prev_record_t = frame["t"]

        now_t = time.time() - t0_wall
        wait_t = cumulative_target_t - now_t

        if wait_t > 0:
            time.sleep(wait_t)

        pos = frame["pos"]
        vel_raw = frame.get("vel", [0.0] * 6)
        # 角速度按该帧的局部有效速度放大
        vel = [v * eff_speed for v in vel_raw]

        gravity = np.array(robot.get_Gravity(), dtype=float)
        friction = friction_compensation(vel)
        torque = limit_torque(gravity + friction)

        robot.pos_vel_tqe_kp_kd(
            pos, vel, torque.tolist(), KP, KD
        )

        if i < RELEASE_FRAME:
            command_gripper(robot, GRIPPER_HOLD_POS)
        else:
            command_gripper(robot, GRIPPER_RELEASE_POS)

        if i % 500 == 0:
            print(f"[Player] frame {i}/{len(frames) - 1}")

    print("[Player] 回放完成")


def main():
    if not os.path.isfile(TRAJECTORY_FILE):
        print(f"文件不存在：{TRAJECTORY_FILE}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")

    robot = Panthera(config_path)

    try:
        frames = load_frames(TRAJECTORY_FILE)
        replay_original_speed(robot, frames, speed=SPEED)

    except KeyboardInterrupt:
        print("\n回放被中断")

    finally:
        print("电机已停止")


if __name__ == "__main__":
    main()