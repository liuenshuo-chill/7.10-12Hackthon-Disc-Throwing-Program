#!/usr/bin/env python3
"""
Record an arm-only teaching trajectory.

This script records only the 6 arm joints. It does not record gripper motion
and does not set the release frame. Use 5_replay_trajectory.py in
MODE="select_release" later to choose the gripper release frame and save a
separate JSONL copy.

新增功能：
    默认全部6个关节都处于零刚度+重力补偿状态，可以自由手动示教。

    按空格键：关节 2/3/4/6（0-based索引 1,2,3,5）切换到"锁定"状态——
        这几个关节会被真正的位置控制（非零kp/kd）锁死在按下那一刻的角度，
        物理上不会再被轻易掰动；写入轨迹文件的数值，也固定成按下那一刻
        的角度/角速度（角速度记为0），不会随任何残余的物理漂移而更新。
        关节 1、5（索引 0, 4）全程不受影响，始终保持零刚度自由示教，
        正常记录真实轨迹。

    再按一次空格：解锁，恢复全部关节自由示教。

按键：
    Space: 切换 锁定/解锁 关节2,3,4,6
    Ctrl+C: 停止并保存
"""
import json
import os
import sys
import select
import termios
import tty
import time

import numpy as np
from Panthera_lib import Panthera

# ---------------- Parameters ----------------
DO_RECORD = True
REC_FILE = None
CONTROL_DT = 0.001

# Leave the gripper in zero-stiffness mode while teaching, but never record it.
FREE_GRIPPER_DURING_TEACHING = False

# 按空格锁定的关节（0-based索引），对应关节2,3,4,6
LOCK_JOINT_INDICES = [1, 2, 3, 5]

# 锁定这些关节时使用的位置控制增益（只有LOCK_JOINT_INDICES对应的位置生效）
LOCK_KP = [0.0, 40.0, 45.0, 20.0, 0.0, 10.0]
LOCK_KD = [0.0, 4.0, 4.5, 2.0, 0.0, 1.0]
# --------------------------------------------


class ArmOnlyTrajectoryRecorder:
    def __init__(self, filepath=None):
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"arm_teach_{timestamp}.jsonl"

        self.filepath = filepath
        self._frames_path = filepath + ".frames.tmp"
        self._fp = open(self._frames_path, "w", encoding="utf-8")
        self.start_time = time.perf_counter()
        self.frame_count = 0

    def log(self, positions, velocities):
        now = time.perf_counter()
        frame = {
            "type": "frame",
            "frame": self.frame_count,
            "t": now - self.start_time,
            "positions": list(map(float, positions)),
            "velocities": list(map(float, velocities)),
        }
        self._fp.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self.frame_count += 1

    def close(self):
        self._fp.close()
        metadata = {
            "type": "metadata",
            "format": "panthera_arm_only_teach_v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": self.frame_count,
            "release_marker": None,
            "notes": "Master trajectory: arm joints only. Use selector mode to create release-marker copies.",
        }

        with open(self.filepath, "w", encoding="utf-8") as out_fp:
            out_fp.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            with open(self._frames_path, "r", encoding="utf-8") as frames_fp:
                for line in frames_fp:
                    out_fp.write(line)

        os.remove(self._frames_path)


# ---------------- 非阻塞终端按键读取（纯终端交互，不用pynput）----------------
_old_term_settings = None


def enable_raw_terminal():
    global _old_term_settings
    fd = sys.stdin.fileno()
    _old_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)


def restore_terminal():
    if _old_term_settings is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, _old_term_settings)


def read_key_nonblocking():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if not dr:
        return None
    return sys.stdin.read(1)
# -----------------------------------------------------------------------


def control_once(lock_state):
    """
    lock_state: dict，包含
        'locked': 当前是否处于锁定状态
        'frozen_pos': 长度6的list，锁定那一刻LOCK_JOINT_INDICES对应关节的角度
    """
    leader_positions = Leader.get_current_pos()
    leader_velocity = Leader.get_current_vel()

    leader_gra = Leader.get_Gravity()
    leader_tor = np.array(leader_gra)

    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    leader_tor = np.clip(leader_tor, -tau_limit, tau_limit)

    # 根据锁定状态，构造这一次下发的position目标和kp/kd
    cmd_pos = list(zero_pos)
    cmd_kp = list(zero_kp)
    cmd_kd = list(zero_kd)

    if lock_state['locked']:
        for j in LOCK_JOINT_INDICES:
            cmd_pos[j] = lock_state['frozen_pos'][j]
            cmd_kp[j] = LOCK_KP[j]
            cmd_kd[j] = LOCK_KD[j]

    # Zero stiffness/zero damping (对未锁定关节) lets the arm be manually guided.
    # 锁定关节则用真实kp/kd做位置控制，锁死在frozen_pos。
    Leader.pos_vel_tqe_kp_kd(cmd_pos, zero_vel, leader_tor, cmd_kp, cmd_kd)

    if FREE_GRIPPER_DURING_TEACHING:
        Leader.gripper_control_MIT(0.0, 0.0, 0.0, 0.0, 0.0)

    # 构造写入文件用的数值：锁定关节写冻结值，其余关节写真实值
    record_positions = list(leader_positions)
    record_velocities = list(leader_velocity)
    if lock_state['locked']:
        for j in LOCK_JOINT_INDICES:
            record_positions[j] = lock_state['frozen_pos'][j]
            record_velocities[j] = 0.0

    print("\r", end="")
    for i in range(Leader.motor_count):
        marker = "L" if (lock_state['locked'] and i in LOCK_JOINT_INDICES) else " "
        print(f"J{i+1}{marker}: {leader_positions[i]:6.3f}rad {leader_velocity[i]:6.3f}rad/s | ", end="")
    frame_text = ""
    if DO_RECORD:
        status = "LOCKED(2346)" if lock_state['locked'] else "free-teach"
        frame_text = f"frames: {rec.frame_count} [{status}]"
    print(frame_text + "   ", end="", flush=True)

    time.sleep(CONTROL_DT)

    return record_positions, record_velocities


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")
    Leader = Panthera(config_path)

    zero_pos = [0.0] * Leader.motor_count
    zero_vel = [0.0] * Leader.motor_count
    zero_kp = [0.0] * Leader.motor_count
    zero_kd = [0.0] * Leader.motor_count

    Fc = np.array([0.15, 0.12, 0.12, 0.12, 0.04, 0.04])
    Fv = np.array([0.05, 0.05, 0.05, 0.03, 0.02, 0.02])
    vel_threshold = 0.02

    if DO_RECORD:
        rec = ArmOnlyTrajectoryRecorder(REC_FILE)
        print(f"Start recording arm-only master trajectory: {rec.filepath}")
        print("This master file will not contain a gripper release marker.")
        print(f"Press SPACE to lock joints {[i+1 for i in LOCK_JOINT_INDICES]} "
              f"(J1 and J5 always stay free).\n")

    lock_state = {'locked': False, 'frozen_pos': [0.0] * Leader.motor_count}

    enable_raw_terminal()

    try:
        for _ in range(10):
            Leader.send_get_motor_state_cmd()
            time.sleep(0.1)

        while True:
            key = read_key_nonblocking()
            if key == ' ':
                lock_state['locked'] = not lock_state['locked']
                if lock_state['locked']:
                    # 刚按下：把当前这一刻的角度锁定下来
                    current = Leader.get_current_pos()
                    lock_state['frozen_pos'] = list(current)

            record_positions, record_velocities = control_once(lock_state)
            if DO_RECORD:
                rec.log(record_positions, record_velocities)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        restore_terminal()
        if DO_RECORD:
            rec.close()
            print(f"Master trajectory saved: {rec.filepath}")
