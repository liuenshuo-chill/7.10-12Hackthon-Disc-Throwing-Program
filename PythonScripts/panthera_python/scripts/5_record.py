#!/usr/bin/env python3
"""
Record an arm-only teaching trajectory.

This script records only the 6 arm joints. It does not record gripper motion
and does not set the release frame. Use 5_replay_trajectory.py in
MODE="select_release" later to choose the gripper release frame and save a
separate JSONL copy.
"""
import json
import os
import time

import numpy as np
from Panthera_lib import Panthera

# ---------------- Parameters ----------------
DO_RECORD = True
REC_FILE = None
CONTROL_DT = 0.001

# Leave the gripper in zero-stiffness mode while teaching, but never record it.
FREE_GRIPPER_DURING_TEACHING = False
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


def control_once():
    leader_positions = Leader.get_current_pos()
    leader_velocity = Leader.get_current_vel()

    leader_gra = Leader.get_Gravity()
    leader_tor = np.array(leader_gra) 
    # + Leader.get_friction_compensation(
    #     leader_velocity, Fc, Fv, vel_threshold
    # )

    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    leader_tor = np.clip(leader_tor, -tau_limit, tau_limit)

    # Zero stiffness/zero damping lets the arm be manually guided.
    Leader.pos_vel_tqe_kp_kd(zero_pos, zero_vel, leader_tor, zero_kp, zero_kd)

    if FREE_GRIPPER_DURING_TEACHING:
        Leader.gripper_control_MIT(0.0, 0.0, 0.0, 0.0, 0.0)

    print("\r", end="")
    for i in range(Leader.motor_count):
        print(f"J{i+1}: {leader_positions[i]:6.3f}rad {leader_velocity[i]:6.3f}rad/s | ", end="")
    frame_text = ""
    if DO_RECORD:
        frame_text = f"frames: {rec.frame_count}"
    print(frame_text + "   ", end="", flush=True)

    time.sleep(CONTROL_DT)


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

    try:
        for _ in range(10):
            Leader.send_get_motor_state_cmd()
            time.sleep(0.1)

        while True:
            control_once()
            if DO_RECORD:
                rec.log(Leader.get_current_pos(), Leader.get_current_vel())

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if DO_RECORD:
            rec.close()
            print(f"Master trajectory saved: {rec.filepath}")
