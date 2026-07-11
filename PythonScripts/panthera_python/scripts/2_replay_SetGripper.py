#!/usr/bin/env python3
"""
Replay arm-only trajectory, or interactively choose a gripper release frame.

MODE="select_release":
    Open a small Tkinter window. Drag the progress bar or press/hold A/D
    or left/right arrows to move through the trajectory. Press SPACE or the
    save button to write a new JSONL copy containing release_marker.

MODE="play":
    Replay the trajectory with real recorded timing. If release_marker exists,
    the gripper opens from that frame/time.
"""
import copy
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from Panthera_lib import Panthera

# ---------------- Parameters ----------------
TRAJECTORY_FILE = "arm_teach_20260711_005433.jsonl"

# "select_release" or "play"
MODE = "select_release"

kp_play = [30.0, 40.0, 55.0, 15.0, 7.0, 5.0]
kd_play = [3.0, 4.0, 5.5, 1.5, 0.7, 0.5]

# Tune these two values on the real gripper.
GRIPPER_HOLD_POS = 0.0
GRIPPER_RELEASE_POS = 1.0
GRIPPER_KP = 5.0
GRIPPER_KD = 0.5

# Selector mode. One tick is one GUI/control update.
SELECTOR_TICK_MS = 20
SELECTOR_MIN_STEP = 1
SELECTOR_MAX_STEP = 50
SELECTOR_DEFAULT_STEP = 5
SELECTOR_USE_RECORDED_VELOCITY = False

Fc = np.array([0.15, 0.12, 0.12, 0.12, 0.04, 0.04])
Fv = np.array([0.05, 0.05, 0.05, 0.03, 0.02, 0.02])
vel_threshold = 0.02
tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
# --------------------------------------------


def load_arm_only_trajectory(filepath):
    metadata = {}
    frames = []

    with open(filepath, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "metadata":
                metadata = item
            elif item.get("type") == "frame":
                frames.append(item)
            else:
                if "positions" in item or "pos" in item:
                    item.setdefault("type", "frame")
                    item.setdefault("frame", len(frames))
                    frames.append(item)

    if not frames:
        raise ValueError("trajectory contains no frames")

    if not metadata:
        metadata = {
            "type": "metadata",
            "format": "panthera_arm_only_teach_v1",
            "release_marker": None,
        }

    return metadata, frames


def read_console_key():
    if os.name != "nt":
        return None

    import msvcrt

    if not msvcrt.kbhit():
        return None

    key = msvcrt.getch()
    if key in (b"p", b"P"):
        return "pause"
    if key in (b"q", b"Q"):
        return "quit"
    return None


def frame_positions(frame):
    return frame.get("positions", frame.get("pos"))


def frame_velocities(frame):
    velocities = frame.get("velocities", frame.get("vel"))
    if velocities is None:
        return [0.0] * len(frame_positions(frame))
    return velocities


def command_frame(robot, frame, use_recorded_velocity=True, hold_gripper=True):
    positions = np.array(frame_positions(frame), dtype=float)
    if use_recorded_velocity:
        velocities = np.array(frame_velocities(frame), dtype=float)
    else:
        velocities = np.zeros(len(positions), dtype=float)

    gravity = np.array(robot.get_Gravity())
    friction = robot.get_friction_compensation(velocities, Fc, Fv, vel_threshold)
    torque = np.clip(gravity + friction, -tau_limit, tau_limit)

    robot.pos_vel_tqe_kp_kd(positions, velocities, torque, kp_play, kd_play)

    if hold_gripper:
        robot.gripper_control_MIT(GRIPPER_HOLD_POS, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)


class ReplayControlWindow:
    def __init__(self):
        self.paused = False
        self.should_quit = False

        self.root = tk.Tk()
        self.root.title("Panthera Replay Control")
        self.root.geometry("360x150")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Playing")
        ttk.Label(main, textvariable=self.status_var, font=("Arial", 12)).pack(fill="x", pady=(0, 12))

        row = ttk.Frame(main)
        row.pack(fill="x")
        self.pause_button = ttk.Button(row, text="Pause (P)", command=self.toggle_pause)
        self.pause_button.pack(side="left", expand=True, fill="x", padx=(0, 8))
        ttk.Button(row, text="Quit (Q)", command=self.quit).pack(side="left", expand=True, fill="x")

        ttk.Label(main, text="Focus this window, then press P/Q.").pack(fill="x", pady=(12, 0))

        self.root.bind("<KeyPress-p>", lambda _event: self.toggle_pause())
        self.root.bind("<KeyPress-P>", lambda _event: self.toggle_pause())
        self.root.bind("<KeyPress-q>", lambda _event: self.quit())
        self.root.bind("<KeyPress-Q>", lambda _event: self.quit())
        self.root.focus_force()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.status_var.set("Paused")
            self.pause_button.configure(text="Resume (P)")
        else:
            self.status_var.set("Playing")
            self.pause_button.configure(text="Pause (P)")

    def quit(self):
        self.should_quit = True

    def poll(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.should_quit = True

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def should_release(frame, index, release_marker):
    if not release_marker:
        return False

    release_frame = release_marker.get("frame")
    release_time = release_marker.get("t")

    if release_frame is not None and index >= int(release_frame):
        return True
    if release_time is not None and float(frame.get("t", 0.0)) >= float(release_time):
        return True
    return False


def output_release_filename(filepath, frame_index):
    root, ext = os.path.splitext(filepath)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{root}_release_f{frame_index}_{timestamp}{ext or '.jsonl'}"


def save_release_copy(filepath, metadata, frames, frame_index):
    selected = frames[frame_index]
    new_metadata = copy.deepcopy(metadata)
    new_metadata["type"] = "metadata"
    new_metadata["format"] = "panthera_arm_only_teach_v1"
    new_metadata["frame_count"] = len(frames)
    new_metadata["release_marker"] = {
        "frame": int(frame_index),
        "t": float(selected.get("t", 0.0)),
    }
    new_metadata["source_file"] = os.path.abspath(filepath)
    new_metadata["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    new_metadata["notes"] = "Copy with gripper release marker. Master trajectory is unchanged."

    out_path = output_release_filename(filepath, frame_index)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(new_metadata, ensure_ascii=False) + "\n")
        for i, frame in enumerate(frames):
            item = dict(frame)
            item["type"] = "frame"
            item["frame"] = i
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    return out_path


class ReleaseSelector:
    def __init__(self, robot, filepath, metadata, frames):
        self.robot = robot
        self.filepath = filepath
        self.metadata = metadata
        self.frames = frames
        self.index = 0
        self.direction = 0
        self.is_dragging = False
        self.is_updating_progress = False

        self.root = tk.Tk()
        self.root.title("Panthera Release Frame Selector")
        self.root.geometry("720x260")

        self.frame_var = tk.IntVar(value=0)
        self.step_var = tk.IntVar(value=SELECTOR_DEFAULT_STEP)
        self.status_var = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
        self._set_index(0, command_robot=True)
        self._tick()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        self.info_label = ttk.Label(main, text="", font=("Arial", 12))
        self.info_label.pack(fill="x", pady=(0, 10))

        self.progress = ttk.Scale(
            main,
            from_=0,
            to=len(self.frames) - 1,
            orient="horizontal",
            command=self._on_progress_drag,
        )
        self.progress.pack(fill="x", pady=(0, 12))
        self.progress.bind("<ButtonPress-1>", self._on_drag_start)
        self.progress.bind("<ButtonRelease-1>", self._on_drag_end)

        row = ttk.Frame(main)
        row.pack(fill="x", pady=(0, 10))

        ttk.Label(row, text="A/Left backward, D/Right forward, Space save").pack(side="left")
        ttk.Button(row, text="Save release copy", command=self._save_current).pack(side="right")

        speed_row = ttk.Frame(main)
        speed_row.pack(fill="x", pady=(0, 10))
        ttk.Label(speed_row, text="Step per tick").pack(side="left")
        self.speed = ttk.Scale(
            speed_row,
            from_=SELECTOR_MIN_STEP,
            to=SELECTOR_MAX_STEP,
            orient="horizontal",
            command=self._on_speed_change,
        )
        self.speed.set(SELECTOR_DEFAULT_STEP)
        self.speed.pack(side="left", fill="x", expand=True, padx=10)
        self.speed_label = ttk.Label(speed_row, text=str(SELECTOR_DEFAULT_STEP), width=4)
        self.speed_label.pack(side="left")

        ttk.Label(main, textvariable=self.status_var).pack(fill="x")

    def _bind_keys(self):
        self.root.bind("<KeyPress-a>", lambda event: self._set_direction(-1))
        self.root.bind("<KeyPress-A>", lambda event: self._set_direction(-1))
        self.root.bind("<KeyPress-Left>", lambda event: self._set_direction(-1))
        self.root.bind("<KeyPress-d>", lambda event: self._set_direction(1))
        self.root.bind("<KeyPress-D>", lambda event: self._set_direction(1))
        self.root.bind("<KeyPress-Right>", lambda event: self._set_direction(1))
        self.root.bind("<KeyRelease-a>", lambda event: self._clear_direction(-1))
        self.root.bind("<KeyRelease-A>", lambda event: self._clear_direction(-1))
        self.root.bind("<KeyRelease-Left>", lambda event: self._clear_direction(-1))
        self.root.bind("<KeyRelease-d>", lambda event: self._clear_direction(1))
        self.root.bind("<KeyRelease-D>", lambda event: self._clear_direction(1))
        self.root.bind("<KeyRelease-Right>", lambda event: self._clear_direction(1))
        self.root.bind("<space>", lambda event: self._save_current())

    def _on_progress_drag(self, value):
        if self.is_updating_progress:
            return
        if self.is_dragging:
            self._set_index(int(float(value)), command_robot=False)

    def _on_drag_start(self, _event):
        self.is_dragging = True
        self.direction = 0

    def _on_drag_end(self, _event):
        self.is_dragging = False
        self._set_index(int(float(self.progress.get())), command_robot=True)

    def _on_speed_change(self, value):
        step = max(SELECTOR_MIN_STEP, int(float(value)))
        self.step_var.set(step)
        self.speed_label.configure(text=str(step))

    def _set_direction(self, direction):
        self.direction = direction

    def _clear_direction(self, direction):
        if self.direction == direction:
            self.direction = 0

    def _set_index(self, index, command_robot=True):
        self.index = max(0, min(len(self.frames) - 1, int(index)))
        frame = self.frames[self.index]
        frame_t = float(frame.get("t", 0.0))

        self.frame_var.set(self.index)
        self.is_updating_progress = True
        self.progress.set(self.index)
        self.is_updating_progress = False
        self.info_label.configure(
            text=f"Frame {self.index} / {len(self.frames) - 1}    t={frame_t:.3f}s"
        )

        marker = self.metadata.get("release_marker")
        if marker:
            self.status_var.set(
                f"Input already has release marker: frame={marker.get('frame')}, t={marker.get('t', 0.0):.3f}s"
            )
        else:
            self.status_var.set("No release marker in input. Save will create a new marked copy.")

        if command_robot:
            command_frame(
                self.robot,
                frame,
                use_recorded_velocity=SELECTOR_USE_RECORDED_VELOCITY,
                hold_gripper=True,
            )

    def _save_current(self):
        out_path = save_release_copy(self.filepath, self.metadata, self.frames, self.index)
        self.status_var.set(f"Saved release copy: {out_path}")
        print(f"\nSaved release copy: {out_path}")

    def _tick(self):
        if self.direction != 0 and not self.is_dragging:
            step = self.step_var.get()
            self._set_index(self.index + self.direction * step, command_robot=True)
        else:
            command_frame(
                self.robot,
                self.frames[self.index],
                use_recorded_velocity=False,
                hold_gripper=True,
            )

        self.root.after(SELECTOR_TICK_MS, self._tick)

    def run(self):
        self.root.mainloop()


def replay(robot, metadata, frames):
    release_marker = metadata.get("release_marker")
    release_started = False

    if release_marker:
        print(
            f"Release marker: frame={release_marker.get('frame')}, "
            f"t={release_marker.get('t', 0.0):.3f}s"
        )
    else:
        print("No release marker found. The gripper will stay at GRIPPER_HOLD_POS.")

    control = ReplayControlWindow()

    try:
        start = time.perf_counter()
        first_t = float(frames[0].get("t", 0.0))

        for index, frame in enumerate(frames):
            target_t = float(frame.get("t", 0.0)) - first_t
            while time.perf_counter() - start < target_t:
                control.poll()
                key = read_console_key()
                if key == "pause":
                    control.toggle_pause()
                elif key == "quit":
                    return

                if control.should_quit:
                    return
                if control.paused:
                    pause_replay(robot, frame, control)
                    start = time.perf_counter() - target_t
                time.sleep(0.0005)

            command_frame(robot, frame, use_recorded_velocity=True, hold_gripper=False)

            if should_release(frame, index, release_marker):
                if not release_started:
                    print(f"Start gripper release at replay frame {index}")
                    release_started = True
                gripper_pos = GRIPPER_RELEASE_POS
            else:
                gripper_pos = GRIPPER_HOLD_POS

            robot.gripper_control_MIT(gripper_pos, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)

            control.poll()
            key = read_console_key()
            if key == "pause":
                control.toggle_pause()
            elif key == "quit":
                return

            if control.should_quit:
                return
            if control.paused:
                pause_replay(robot, frame, control)
                start = time.perf_counter() - target_t
    finally:
        control.close()


def pause_replay(robot, frame, control):
    print("\nReplay paused. Press P in the control window to resume, Q to quit.")
    while control.paused and not control.should_quit:
        command_frame(robot, frame, use_recorded_velocity=False, hold_gripper=False)
        robot.gripper_control_MIT(GRIPPER_HOLD_POS, 0.0, 0.0, GRIPPER_KP, GRIPPER_KD)

        control.poll()
        key = read_console_key()
        if key == "pause":
            control.toggle_pause()
        if key == "quit":
            control.quit()

        time.sleep(0.01)

    if control.should_quit:
        print("Replay quit.")
    else:
        print("Replay resumed.")


if __name__ == "__main__":
    if not os.path.isfile(TRAJECTORY_FILE):
        print(f"File not found: {TRAJECTORY_FILE}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")
    robot = Panthera(config_path)

    try:
        metadata, frames = load_arm_only_trajectory(TRAJECTORY_FILE)
        print(f"Trajectory: {TRAJECTORY_FILE}")
        print(f"Frames: {len(frames)}")

        if MODE == "select_release":
            selector = ReleaseSelector(robot, TRAJECTORY_FILE, metadata, frames)
            selector.run()
        elif MODE == "play":
            replay(robot, metadata, frames)
        else:
            raise ValueError('MODE must be "select_release" or "play"')

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Finished.")
