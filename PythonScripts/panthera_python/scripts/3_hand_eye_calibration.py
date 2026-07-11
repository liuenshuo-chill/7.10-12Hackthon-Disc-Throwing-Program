#!/usr/bin/env python3
"""
Eye-in-Hand 标定程序（手动拖动模式 + RealSense D405）

功能：
    1. 手动拖动机械臂到不同姿态
    2. 在每个姿态下按空格键采集数据（机械臂位姿 + 相机观测）
    3. 采集足够数据后，使用 Hand-Eye 标定算法求解相机到末端的变换矩阵
    4. 保存标定结果

使用步骤：
    1. 将标定板（ArUco 标记）固定在工作空间中
    2. 运行程序
    3. 手动拖动机械臂到不同姿态（确保相机能看到标定板）
    4. 在每个姿态下按空格键采集数据
    5. 采集 10-20 组数据后按 'C' 进行标定
    6. 标定结果自动保存到文件

注意事项：
    - 程序不会自动控制机械臂运动
    - 需要手动拖动机械臂到不同姿态
    - 姿态要有足够的多样性（不同位置、不同角度）
    - 每个姿态都要确保标定板在相机视野内
    - D405 最佳工作距离：7-50cm

依赖：
    pip install opencv-contrib-python pyrealsense2
"""

import time
import numpy as np
import cv2
import json
import pyrealsense2 as rs
from Panthera_lib import Panthera
from pynput import keyboard

# 全局变量
robot_poses = []  # 机械臂末端位姿列表 (base -> TCP)
camera_poses = []  # 相机观测到的标定板位姿列表 (camera -> marker)
current_mode = "wait"  # wait / collect / calibrate / done
running = True
pipeline = None
align = None

# 标定板参数
CALIBRATION_PATTERN = "chessboard"  # "chessboard" 或 "aruco"
# 如果棋盘格坐标系跳变，可以改为 "aruco" 使用 ArUco 标记

# 棋盘格参数
CHESSBOARD_SIZE = (7, 7)  # 内角点数量 (8x8棋盘格有7x7个内角点)
SQUARE_SIZE = 0.020  # 方格尺寸（米）20mm = 0.020m

# ArUco 参数
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE = 0.05  # ArUco标记尺寸（米），根据实际打印尺寸修改

# 是否启用角点一致性检查（仅用于棋盘格）
CHECK_CORNER_CONSISTENCY = False
first_corner_position = None  # 记录第一次采集的角点位置
CAMERA_MATRIX = None  # 相机内参矩阵
DIST_COEFFS = None  # 相机畸变系数


def init_realsense():
    """
    初始化 RealSense D405 相机

    返回：
        pipeline: RealSense pipeline
        align: 对齐对象（将深度对齐到彩色）
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    print("初始化 RealSense D405 相机...")

    # 创建 pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置流
    # D405 推荐配置：640x480 @ 30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动 pipeline
    profile = pipeline.start(config)

    # 获取相机内参
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # 转换为 OpenCV 格式
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    # D405 畸变系数（Brown-Conrady 模型）
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

    # 创建对齐对象
    align = rs.align(rs.stream.color)

    print(f"✓ 相机初始化成功")
    print(f"  分辨率: {intrinsics.width}x{intrinsics.height}")
    print(f"  焦距: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print(f"  主点: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    # 预热相机（丢弃前几帧）
    print("预热相机...")
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, camera_matrix, dist_coeffs


def get_realsense_frame(pipeline, align):
    """
    获取 RealSense 相机图像

    参数：
        pipeline: RealSense pipeline
        align: 对齐对象

    返回：
        color_image: 彩色图像
        depth_image: 深度图像（对齐到彩色）
    """
    # 等待帧
    frames = pipeline.wait_for_frames()

    # 对齐深度到彩色
    aligned_frames = align.process(frames)

    # 获取对齐后的帧
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None

    # 转换为 numpy 数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image


def check_corner_consistency(corners, tolerance=50):
    """
    检查角点位置是否与第一次采集一致

    参数：
        corners: 当前检测到的角点
        tolerance: 允许的像素偏差（像素）

    返回：
        is_consistent: 是否一致
        warning_message: 警告信息
    """
    global first_corner_position

    # 获取第一个角点的位置
    current_first_corner = corners[0].ravel()

    if first_corner_position is None:
        # 第一次采集，记录角点位置
        first_corner_position = current_first_corner.copy()
        return True, None

    # 计算与第一次采集的偏差
    distance = np.linalg.norm(current_first_corner - first_corner_position)

    if distance > tolerance:
        warning_message = f"警告：原点位置偏差 {distance:.1f} 像素（阈值 {tolerance}）"
        return False, warning_message

    return True, None


def detect_chessboard(image, camera_matrix, dist_coeffs):
    """
    检测棋盘格并估计位姿

    参数：
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数

    返回：
        success: 是否检测成功
        rvec: 旋转向量
        tvec: 平移向量
        annotated_image: 标注后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 复制图像用于标注
    annotated_image = image.copy()

    # 查找棋盘格角点
    # 使用 CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 提高检测稳定性
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 检查角点一致性
        if CHECK_CORNER_CONSISTENCY:
            is_consistent, warning_msg = check_corner_consistency(corners_refined)
            if not is_consistent:
                # 角点位置不一致，标记为红色
                cv2.drawChessboardCorners(annotated_image, CHESSBOARD_SIZE, corners_refined, ret)
                cv2.putText(annotated_image, warning_msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, "INCONSISTENT ORIGIN!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return False, None, None, annotated_image

        # 绘制角点
        cv2.drawChessboardCorners(annotated_image, CHESSBOARD_SIZE, corners_refined, ret)

        # 构建3D点（棋盘格在世界坐标系中的位置）
        # 重要：确保坐标系原点始终在同一个角点
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE

        # 估计位姿
        success, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

        if success:
            # 绘制坐标轴（原点在第一个角点）
            axis_length = SQUARE_SIZE * 3
            cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

            # 标记原点位置（第一个角点）
            origin_corner = tuple(corners_refined[0].ravel().astype(int))
            cv2.circle(annotated_image, origin_corner, 10, (0, 0, 255), -1)  # 红色圆点
            cv2.putText(annotated_image, "Origin", (origin_corner[0] + 15, origin_corner[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # 显示距离信息
            distance = np.linalg.norm(tvec)
            text = f"Distance: {distance*1000:.1f}mm"
            cv2.putText(annotated_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示角点顺序提示
            cv2.putText(annotated_image, f"Corners: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            return True, rvec, tvec, annotated_image
        else:
            cv2.putText(annotated_image, "Pose estimation failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return False, None, None, annotated_image
    else:
        # 未检测到棋盘格
        cv2.putText(annotated_image, "No chessboard detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return False, None, None, annotated_image


def detect_calibration_pattern(image, camera_matrix, dist_coeffs):
    """
    检测标定板（自动选择棋盘格或ArUco）

    参数：
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数

    返回：
        success: 是否检测成功
        rvec: 旋转向量
        tvec: 平移向量
        annotated_image: 标注后的图像
    """
    if CALIBRATION_PATTERN == "chessboard":
        return detect_chessboard(image, camera_matrix, dist_coeffs)
    elif CALIBRATION_PATTERN == "aruco":
        return detect_aruco_marker(image, camera_matrix, dist_coeffs)
    else:
        raise ValueError(f"未知的标定板类型: {CALIBRATION_PATTERN}")
    """
    检测 ArUco 标记并估计位姿

    参数：
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数

    返回：
        success: 是否检测成功
        rvec: 旋转向量
        tvec: 平移向量
        annotated_image: 标注后的图像
    """
    # 创建 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 复制图像用于标注
    annotated_image = image.copy()

    # 检测标记
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None and len(ids) > 0:
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)

        # 估计位姿
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )

        # 使用第一个检测到的标记
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        # 绘制坐标轴
        cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)

        # 在图像上显示距离信息
        distance = np.linalg.norm(tvec)
        text = f"Distance: {distance*1000:.1f}mm"
        cv2.putText(annotated_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return True, rvec, tvec, annotated_image
    else:
        # 在图像上显示未检测到标记
        cv2.putText(annotated_image, "No marker detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return False, None, None, annotated_image


def rvec_tvec_to_matrix(rvec, tvec):
    """
    将旋转向量和平移向量转换为 4x4 变换矩阵

    参数：
        rvec: 旋转向量
        tvec: 平移向量

    返回：
        T: 4x4 变换矩阵
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()  # 展平为一维数组
    return T


def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角（ZYX 顺序）

    参数：
        R: 3x3 旋转矩阵

    返回：
        roll, pitch, yaw: 欧拉角（弧度）
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def get_robot_tcp_pose(robot):
    """
    获取机械臂末端位姿（base -> TCP）

    参数：
        robot: Panthera 机器人实例

    返回：
        T: 4x4 变换矩阵
    """
    fk = robot.forward_kinematics()
    position = fk['position']
    rotation = fk['rotation']

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position

    return T


def collect_calibration_data(robot, image):
    """
    采集一组标定数据

    参数：
        robot: Panthera 机器人实例
        image: 当前相机图像

    返回：
        success: 是否采集成功
        annotated_image: 标注后的图像（用于显示）
    """
    global robot_poses, camera_poses

    print("  - 检测标定板...")
    # 检测标定板
    success, rvec, tvec, annotated_image = detect_calibration_pattern(image, CAMERA_MATRIX, DIST_COEFFS)

    if not success:
        if CALIBRATION_PATTERN == "chessboard" and CHECK_CORNER_CONSISTENCY:
            print("  ✗ 标定板检测失败或原点位置不一致")
            print("  提示：确保棋盘格方向与第一次采集时相同")
        else:
            print("  ✗ 未检测到标定板，请调整机械臂姿态")
        return False, annotated_image

    print("  ✓ 标定板检测成功")
    print("  - 读取机械臂位姿...")

    # 获取机械臂末端位姿
    T_base_tcp = get_robot_tcp_pose(robot)

    # 获取相机观测到的标定板位姿
    T_camera_marker = rvec_tvec_to_matrix(rvec, tvec)

    # 保存数据
    robot_poses.append(T_base_tcp)
    camera_poses.append(T_camera_marker)

    print(f"  ✓ 第 {len(robot_poses)} 组数据已保存")
    print(f"    机械臂位置: [{T_base_tcp[0,3]:.3f}, {T_base_tcp[1,3]:.3f}, {T_base_tcp[2,3]:.3f}]")
    print(f"    标记距离: {np.linalg.norm(tvec)*1000:.1f} mm")

    return True, annotated_image


def perform_hand_eye_calibration():
    """
    执行 Hand-Eye 标定

    返回：
        success: 是否标定成功
        T_tcp_camera: TCP 到相机的变换矩阵
    """
    global robot_poses, camera_poses

    if len(robot_poses) < 3:
        print("数据不足，至少需要 3 组数据")
        return False, None

    print(f"\n开始标定，共 {len(robot_poses)} 组数据...")

    # 准备数据：计算相邻姿态之间的变换
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(robot_poses)):
        # 机械臂末端位姿 (base -> TCP)
        T_base_tcp = robot_poses[i]
        R_gripper2base.append(T_base_tcp[:3, :3])
        t_gripper2base.append(T_base_tcp[:3, 3])

        # 相机观测到的标定板位姿 (camera -> marker)
        T_camera_marker = camera_poses[i]
        R_target2cam.append(T_camera_marker[:3, :3])
        t_target2cam.append(T_camera_marker[:3, 3])

    # 使用 OpenCV 的 Hand-Eye 标定
    try:
        R_tcp_camera, t_tcp_camera = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构建变换矩阵
        T_tcp_camera = np.eye(4)
        T_tcp_camera[:3, :3] = R_tcp_camera
        T_tcp_camera[:3, 3] = t_tcp_camera.flatten()

        print("\n✓ 标定成功！")
        print("\nTCP 到相机的变换矩阵 (T_tcp_camera):")
        print(T_tcp_camera)

        # 转换为欧拉角
        roll, pitch, yaw = rotation_matrix_to_euler(R_tcp_camera)
        print(f"\n位置 (m): [{t_tcp_camera[0,0]:.4f}, {t_tcp_camera[1,0]:.4f}, {t_tcp_camera[2,0]:.4f}]")
        print(f"姿态 (度): [roll={np.rad2deg(roll):.2f}, pitch={np.rad2deg(pitch):.2f}, yaw={np.rad2deg(yaw):.2f}]")

        return True, T_tcp_camera

    except Exception as e:
        print(f"\n✗ 标定失败: {e}")
        return False, None


def save_calibration_result(T_tcp_camera, file_path):
    """
    保存标定结果

    参数：
        T_tcp_camera: TCP 到相机的变换矩阵
        file_path: 保存文件路径
    """
    result = {
        'T_tcp_camera': T_tcp_camera.tolist(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(robot_poses)
    }

    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n标定结果已保存到: {file_path}")


def on_press(key):
    """键盘按下事件处理"""
    global current_mode, running, first_corner_position

    # 调试：打印所有按键
    print(f"[DEBUG] 检测到按键: {key}")

    # 处理特殊键（空格、ESC）
    if key == keyboard.Key.space:
        # 采集数据
        current_mode = "collect"
        print("\n[按下空格键] 准备采集数据...")
        return
    elif key == keyboard.Key.esc:
        # 退出
        print("\n[按下ESC键] 准备退出...")
        running = False
        return False

    # 处理字符键（C、R、S）
    try:
        if hasattr(key, 'char') and key.char:
            if key.char == 'c' or key.char == 'C':
                # 开始标定
                current_mode = "calibrate"
                print("\n[按下C键] 开始标定...")
            elif key.char == 'r' or key.char == 'R':
                # 重置数据
                global robot_poses, camera_poses
                robot_poses = []
                camera_poses = []
                first_corner_position = None  # 重置角点记录
                print("\n[按下R键] 数据已重置（包括角点参考位置）")
            elif key.char == 's' or key.char == 'S':
                # 备用采集键（S键）
                current_mode = "collect"
                print("\n[按下S键] 准备采集数据...")
    except AttributeError:
        pass


def main():
    """
    主函数
    """
    global CAMERA_MATRIX, DIST_COEFFS, current_mode, running, pipeline, align

    print("="*60)
    print("Eye-in-Hand 标定程序 (RealSense D405)")
    print("="*60)

    # 初始化 RealSense 相机
    try:
        pipeline, align, CAMERA_MATRIX, DIST_COEFFS = init_realsense()
    except Exception as e:
        print(f"相机初始化失败: {e}")
        print("请检查相机连接")
        return

    # 初始化机械臂
    print("\n初始化机械臂...")
    robot = Panthera()

    print("\n" + "="*60)
    print("操作说明：")
    print("  1. 手动拖动机械臂到不同姿态")
    print("     - 确保棋盘格在相机视野内")
    print("     - 棋盘格：8x8，方格尺寸20mm")
    print("     - 姿态要有多样性（不同位置、不同角度）")
    print("     - D405 最佳距离：7-50cm")
    print("  2. 按 S 键或空格键采集当前姿态的数据")
    print("  3. 采集至少 10 组数据（建议 15-20 组）")
    print("  4. 按 'C' 键开始标定")
    print("  5. 按 'R' 键重置数据")
    print("  6. 按 ESC 键退出")
    print("="*60)
    print("\n重要提示（防止坐标系跳变）：")
    print("  ✓ 将棋盘格固定在桌面上，不要移动或旋转")
    print("  ✓ 只移动机械臂和相机，保持棋盘格静止")
    print("  ✓ 程序会自动检查原点位置一致性")
    print("  ✓ 如果检测到原点跳变，会拒绝采集并显示红色警告")
    print("  ✓ 看到警告时，按 R 键重置后重新开始")
    print("="*60)
    print("\n准备就绪，请手动拖动机械臂并采集数据...")
    print("提示：如果空格键无响应，请使用 S 键采集数据\n")

    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 创建显示窗口
    cv2.namedWindow('RealSense D405 - Hand-Eye Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RealSense D405 - Hand-Eye Calibration', 640, 480)

    print("✓ 重力补偿已启用，可以轻松拖动机械臂\n")

    try:
        while running:
            # 获取当前关节角度
            q = robot.get_current_pos()

            # 计算重力补偿力矩
            gra = robot.get_Gravity(q)

            # 发送重力补偿控制命令（MIT模式）
            robot.pos_vel_tqe_kp_kd(
                [0.0] * robot.motor_count,  
                [0.0] * robot.motor_count,  
                gra,                        # 重力补偿力矩
                [0.0] * robot.motor_count,                     
                [0.0] * robot.motor_count                   
            )

            # 获取相机图像
            color_image, depth_image = get_realsense_frame(pipeline, align)

            if color_image is None:
                print("无法获取相机图像")
                time.sleep(0.1)
                continue

            # 检测标定板（实时显示）
            _, _, _, annotated_image = detect_calibration_pattern(color_image, CAMERA_MATRIX, DIST_COEFFS)

            # 在图像上显示采集数量和当前模式
            text = f"Samples: {len(robot_poses)}"
            cv2.putText(annotated_image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)

            # 显示当前模式
            mode_text = f"Mode: {current_mode}"
            cv2.putText(annotated_image, mode_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # 显示图像
            cv2.imshow('RealSense D405 - Hand-Eye Calibration', annotated_image)

            # 处理按键（不阻塞）
            cv2.waitKey(1)

            if current_mode == "collect":
                # 采集数据
                print(f"\n正在采集第 {len(robot_poses)+1} 组数据...")
                success, _ = collect_calibration_data(robot, color_image)
                if success:
                    print("✓ 采集成功")
                else:
                    print("✗ 采集失败")
                current_mode = "wait"

            elif current_mode == "calibrate":
                # 执行标定
                success, T_tcp_camera = perform_hand_eye_calibration()

                if success:
                    # 保存结果
                    save_calibration_result(T_tcp_camera, "hand_eye_calibration.json")
                    current_mode = "done"
                else:
                    current_mode = "wait"

            elif current_mode == "done":
                print("\n标定完成！按 ESC 退出")
                time.sleep(1)
                current_mode = "wait"

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n程序被中断")
    finally:
        listener.stop()
        cv2.destroyAllWindows()
        if pipeline:
            pipeline.stop()
        print("程序退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
