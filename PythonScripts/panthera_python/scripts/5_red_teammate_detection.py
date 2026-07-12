#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接盘人识别模块——独立测试脚本（不初始化机械臂，只测感知这一块）

===========================================================================
写这个脚本的目的
===========================================================================
你反馈"放一个纯红色盘在前面也被识别成目标了，没管举手"——这说明整条判断
链路（YOLO人体检测 -> 红衣占比 -> 举手判断）里，至少有一环给出了不该给的
"通过"。但光看最终结果（认/不认）看不出到底是哪一环出的问题，所以这个脚本
把 detect_teammate() 内部每一步的中间值都单独打印/画出来，逐帧展示：

    - 这一帧 YOLO 到底检测到几个"人"，每个人的检测框置信度是多少
    - 每个人的躯干ROI框画在哪、红色占比具体是多少（跟阈值比较的结果）
    - 每个人的左右手腕/肩膀关键点坐标和置信度分别是多少
    - 举手判断具体在比较哪两个数字、结果是 True 还是 False
    - 最终这个人有没有被选中当目标，如果被选中，是上面哪几层判断都通过了

===========================================================================
可能的问题排查方向（供你对照实测结果判断）
===========================================================================
你描述的现象是"只有红盘，没有人举手，却被认成目标"，结合代码逻辑，最可能
出问题的地方有几个（这个测试脚本会把对应的中间值都打出来，跑一遍就知道
是不是这几个原因）：

  1. 画面里其实有一个真人（比如你自己拿着红盘），YOLO 检测到了这个人，
     红盘刚好挡在或者贴近躯干附近，导致 red_ratio_in_roi 算出来的红色占比
     超过了阈值——这种情况下"红衣判断"本身没错，错的是你以为只有盘、其实
     画面里还有人。脚本会把每个人的躯干ROI框画出来，一眼能看出红色占比是
     不是被盘"蹭"上去的。

  2. valid_kpt() 的置信度阈值 KEYPOINT_CONF_THRES 设得太低，导致低质量、
     不可信的关键点也被当成"有效"参与举手判断，产生误判。脚本会把每个
     关键点的实际置信度数值打印出来，能看出是不是卡在阈值边缘。

  3. hand_raise_from_keypoints() 只要求"手腕y坐标比肩膀y坐标小超过
     HAND_RAISE_MARGIN_PX"，如果一个人整个上半身姿态特殊（比如弯腰、
     侧躺、被遮挡导致关键点错位），可能在没有主观"举手"的情况下也满足
     这个像素几何关系。脚本会把具体参与比较的两个y坐标数值打出来。

  4. 如果画面里确实没有任何人形（只有红盘本身），但 YOLO 依然给出了一个
     "person"检测框（模型误检，红盘的形状/颜色触发了假阳性）——这种情况
     该检测框大概率没有可信的关键点（wrist/shoulder 置信度会很低甚至是
     0），脚本能看出这种"框在但关键点全部无效"的情况，如果出现这种情况，
     说明是 YOLO 本身的误检，需要考虑提高 PERSON_CONF_THRES 或者换模型。

===========================================================================
运行逻辑 / 工作流
===========================================================================
主循环每一帧：
    1. 读取相机彩色帧
    2. model.predict() 跑 YOLO-Pose，拿到这一帧所有"人"的检测框+关键点
    3. preprocess_red_mask() 生成全图红色掩膜（跟正式脚本用的是同一份代码）
    4. 对每一个检测框依次算：
         torso_roi_from_keypoints_or_bbox() -> red_ratio_in_roi()
         hand_raise_from_keypoints()
       并且不像正式脚本那样"红衣不过关就 continue 直接跳过"——这里无论
       过不过关都继续往下算、都画出来，因为测试阶段更需要看到"没通过的
       原因"，而不是只看到通过的结果
    5. 把每个人的完整诊断信息画在画面上、同时在终端打印一份文字版
    6. 按当前的判断标准（红衣过关 且 举手）决定要不要在画面上标成"TARGET"，
       但这只是最后展示用，不会对机械臂做任何动作——这个脚本从头到尾不
       import、不初始化 Panthera，纯感知验证。

按 q 退出，按 m 切换是否叠加显示红色掩膜。
===========================================================================
"""

import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# =========================================================================
# 配置（跟正式脚本 scan_and_aim_throw_reverse90.py 保持一致，方便对照）
# =========================================================================
MODEL_PATH = r"/home/laven/Panthera_HT_SDK_Extensions/Panthera-HT_vision_servo_sdk/panthera_python/scripts/yolov8n-pose.pt"

WIDTH, HEIGHT, FPS = 640, 480, 30
MIN_DEPTH_MM = 70
MAX_DEPTH_MM = 3000

PERSON_CONF_THRES = 0.35
KEYPOINT_CONF_THRES = 0.35
RED_RATIO_THRES = 0.08
HAND_RAISE_MARGIN_PX = 20

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

RED_LOWER_1 = np.array([0, 80, 50]);   RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 80, 50]); RED_UPPER_2 = np.array([180, 255, 255])

TORSO_EXPAND = 0.18


# =========================================================================
# 以下函数跟正式脚本完全一致（照搬过来，保证测试结果跟正式运行时一致）
# =========================================================================
def init_realsense():
    # 创建 RealSense 数据管线
    pipeline = rs.pipeline()

    # 创建相机配置对象
    config = rs.config()

    # 启用深度流：分辨率 WIDTH x HEIGHT，格式 z16，帧率 FPS
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    # 启用彩色图像流：BGR 格式，方便 OpenCV 处理
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 启动相机
    profile = pipeline.start(config)

    # 获取深度传感器，用于读取深度比例尺
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 获取彩色相机内参，后续如果需要像素坐标转空间坐标会用到
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsic = color_profile.get_intrinsics()

    # 创建对齐对象：将深度图对齐到彩色图坐标系
    align = rs.align(rs.stream.color)

    # 预热相机，丢弃前 30 帧不稳定画面
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, intrinsic, depth_scale


def preprocess_red_mask(color_image):
    # 使用双边滤波降噪，同时尽量保留边缘
    blurred = cv2.bilateralFilter(color_image, d=7, sigmaColor=60, sigmaSpace=60)

    # 将 BGR 图像转换到 HSV 色彩空间，便于提取红色区域
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 红色在 HSV 中跨越 0 度边界，所以通常分成两个范围提取
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)

    # 合并两个红色掩膜
    mask = cv2.bitwise_or(mask1, mask2)

    # 构造椭圆形形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 开运算：去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算：填补红色区域中的小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def clip_bbox(x1, y1, x2, y2):
    # 将检测框坐标限制在图像范围内，防止数组越界
    x1 = max(0, min(WIDTH - 1, int(x1)))
    y1 = max(0, min(HEIGHT - 1, int(y1)))
    x2 = max(0, min(WIDTH - 1, int(x2)))
    y2 = max(0, min(HEIGHT - 1, int(y2)))

    return x1, y1, x2, y2


def valid_kpt(kpts, idx):
    # 判断某个关键点是否存在，并且置信度是否达到阈值
    return (
        kpts is not None
        and idx < len(kpts)
        and float(kpts[idx][2]) >= KEYPOINT_CONF_THRES
    )


def torso_roi_from_keypoints_or_bbox(kpts, bbox):
    x1, y1, x2, y2 = bbox

    # 躯干区域需要用到左右肩、左右髋四个关键点
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    # 如果四个关键点都有效，就优先用关键点计算躯干 ROI
    if all(valid_kpt(kpts, i) for i in needed):
        xs = [float(kpts[i][0]) for i in needed]
        ys = [float(kpts[i][1]) for i in needed]

        # 根据关键点求出最小外接矩形
        rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)

        # 对躯干区域适当扩张，提高容错性
        w, h = rx2 - rx1, ry2 - ry1
        rx1 -= w * TORSO_EXPAND
        rx2 += w * TORSO_EXPAND
        ry1 -= h * TORSO_EXPAND
        ry2 += h * TORSO_EXPAND

        return clip_bbox(rx1, ry1, rx2, ry2), True

    # 如果关键点不完整，则退化为使用人体框中上半部分作为躯干区域
    return clip_bbox(
        x1,
        y1 + 0.15 * (y2 - y1),
        x2,
        y1 + 0.70 * (y2 - y1)
    ), False


def red_ratio_in_roi(red_mask, roi):
    x1, y1, x2, y2 = roi

    # 防止无效 ROI
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # 截取 ROI 区域内的红色掩膜
    patch = red_mask[y1:y2, x1:x2]

    if patch.size == 0:
        return 0.0

    # 计算 ROI 中红色像素占比
    return float(np.count_nonzero(patch)) / float(patch.size)


def hand_raise_diag(kpts):
    """
    判断是否举手，并返回左右手的诊断信息。
    判断依据：
    如果手腕 y 坐标明显高于肩膀 y 坐标，则认为该侧手臂举起。
    注意：图像坐标系中，y 越小表示位置越靠上。
    """
    diag = {"left": None, "right": None}

    # 判断左手是否举起
    if valid_kpt(kpts, LEFT_WRIST) and valid_kpt(kpts, LEFT_SHOULDER):
        wrist_y = float(kpts[LEFT_WRIST][1])
        shoulder_y = float(kpts[LEFT_SHOULDER][1])

        # 手腕高于肩膀一定像素距离，则认为左手举起
        raised = wrist_y < shoulder_y - HAND_RAISE_MARGIN_PX

        diag["left"] = {
            "wrist_conf": float(kpts[LEFT_WRIST][2]),
            "shoulder_conf": float(kpts[LEFT_SHOULDER][2]),
            "wrist_y": wrist_y,
            "shoulder_y": shoulder_y,
            "raised": raised,
        }

    # 判断右手是否举起
    if valid_kpt(kpts, RIGHT_WRIST) and valid_kpt(kpts, RIGHT_SHOULDER):
        wrist_y = float(kpts[RIGHT_WRIST][1])
        shoulder_y = float(kpts[RIGHT_SHOULDER][1])

        # 手腕高于肩膀一定像素距离，则认为右手举起
        raised = wrist_y < shoulder_y - HAND_RAISE_MARGIN_PX

        diag["right"] = {
            "wrist_conf": float(kpts[RIGHT_WRIST][2]),
            "shoulder_conf": float(kpts[RIGHT_SHOULDER][2]),
            "wrist_y": wrist_y,
            "shoulder_y": shoulder_y,
            "raised": raised,
        }

    # 只要左手或右手任意一只手举起，就认为该人举手
    hand_raised = bool(
        (diag["left"] and diag["left"]["raised"])
        or (diag["right"] and diag["right"]["raised"])
    )

    return hand_raised, diag


# =========================================================================
# 主循环：读取相机画面 -> 检测人体 -> 判断红色衣服 -> 判断举手 -> 可视化结果
# =========================================================================
def main():
    # 重点：程序启动前先检查 YOLO-Pose 模型文件是否存在
    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    # 重点：初始化 RealSense 相机，获取图像流、对齐器、相机内参和深度比例
    print("初始化 RealSense 相机...")
    pipeline, align, intrinsic, depth_scale = init_realsense()

    # 重点：加载 YOLO-Pose 模型，用于人体检测和关键点识别
    print("加载 YOLO-Pose 模型...")
    model = YOLO(MODEL_PATH)

    # 创建 OpenCV 显示窗口
    cv2.namedWindow("Teammate Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Teammate Detection Test", WIDTH, HEIGHT)

    # 是否显示红色掩膜，默认不显示
    show_mask = False

    print("\n" + "=" * 60)
    print("按 q 退出，按 m 切换红色掩膜显示")
    print("=" * 60 + "\n")

    try:
        # 重点：主循环，每次循环处理一帧相机画面
        while True:
            # 重点：从 RealSense 获取一组新的帧
            frames = pipeline.wait_for_frames()

            # 重点：将深度帧对齐到彩色帧坐标系
            aligned = align.process(frames)

            # 获取彩色图像帧
            color_frame = aligned.get_color_frame()

            # 如果没有获取到彩色帧，则跳过本次循环
            if not color_frame:
                continue

            # 重点：将 RealSense 图像帧转换为 OpenCV 可处理的 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 重点：使用 YOLO-Pose 对当前画面进行人体检测和姿态关键点检测
            result = model.predict(
                color_image,
                conf=PERSON_CONF_THRES,
                verbose=False
            )[0]

            # 重点：提取当前画面中的红色区域，生成红色掩膜
            red_mask = preprocess_red_mask(color_image)

            # 创建显示图像副本，后续在上面画框和文字
            display = color_image.copy()

            # 获取检测框和人体关键点结果
            boxes = result.boxes
            keypoints = result.keypoints

            print(
                f"\n--- 新一帧 --- 检测到 "
                f"{0 if boxes is None else len(boxes)} 个候选人体框"
            )

            # 重点：只有同时检测到人体框和关键点时，才进行后续判断
            if boxes is not None and keypoints is not None:
                # 将检测框、置信度、关键点从 GPU/Tensor 转为 numpy，便于处理
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                kpts_all = keypoints.data.cpu().numpy()

                # 重点：逐个处理每一个检测到的人
                for i, (box, conf) in enumerate(zip(xyxy, confs)):
                    # 修正人体框坐标，防止越界
                    bbox = clip_bbox(*box)

                    # 获取当前人的关键点
                    kpts = kpts_all[i] if i < len(kpts_all) else None

                    # 重点：计算该人的躯干 ROI
                    # 优先用肩膀和髋部关键点；关键点不可靠时退化为人体框中上区域
                    torso_roi, torso_from_kpts = torso_roi_from_keypoints_or_bbox(
                        kpts,
                        bbox
                    )

                    # 重点：计算躯干 ROI 内红色像素比例
                    red_ratio = red_ratio_in_roi(red_mask, torso_roi)

                    # 重点：判断红色比例是否超过阈值，用于判断是否穿红色衣服
                    red_pass = red_ratio >= RED_RATIO_THRES

                    # 重点：判断该人是否举手，并返回左右手判断细节
                    hand_raised, hand_diag = hand_raise_diag(kpts)

                    # 重点：最终目标判断条件
                    # 1. 人体检测置信度足够高
                    # 2. 躯干区域红色比例达到阈值
                    # 3. 至少一只手处于举起状态
                    is_target = (
                        float(conf) >= PERSON_CONF_THRES
                        and red_pass
                        and hand_raised
                    )

                    # 重点：在终端输出当前人的完整诊断信息，方便调试阈值
                    print(
                        f"[人{i}] det_conf={conf:.2f}(阈值{PERSON_CONF_THRES}) "
                        f"| 躯干ROI来源={'关键点' if torso_from_kpts else 'bbox退化'} "
                        f"| red_ratio={red_ratio:.3f}(阈值{RED_RATIO_THRES}) "
                        f"{'PASS' if red_pass else 'FAIL'}"
                    )

                    # 输出左右手举手判断细节
                    for side in ("left", "right"):
                        d = hand_diag[side]

                        if d is None:
                            print(
                                f"      {side}手 关键点置信度不足"
                                f"(<{KEYPOINT_CONF_THRES})，跳过判断"
                            )
                        else:
                            print(
                                f"      {side}手 wrist_y={d['wrist_y']:.1f} vs "
                                f"shoulder_y={d['shoulder_y']:.1f}"
                                f"-{HAND_RAISE_MARGIN_PX} "
                                f"| wrist_conf={d['wrist_conf']:.2f} "
                                f"shoulder_conf={d['shoulder_conf']:.2f} "
                                f"| raised={d['raised']}"
                            )

                    print(
                        f"      => hand_raised={hand_raised} | 最终判定 "
                        f"{'*** TARGET ***' if is_target else 'not target'}"
                    )

                    # 重点：以下为画面可视化部分
                    x1, y1, x2, y2 = bbox

                    # 如果是目标人物，用绿色框；否则用红色框
                    box_color = (0, 255, 0) if is_target else (0, 0, 255)

                    # 绘制人体检测框
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                    # 绘制躯干 ROI 区域
                    tx1, ty1, tx2, ty2 = torso_roi
                    cv2.rectangle(display, (tx1, ty1), (tx2, ty2), (255, 200, 0), 1)

                    # 构造显示标签：检测置信度、红色比例、红色是否通过、是否举手
                    label = (
                        f"conf{conf:.2f} red{red_ratio:.2f}"
                        f"{'P' if red_pass else 'F'} "
                        f"hand{'T' if hand_raised else 'F'}"
                    )

                    # 在人体框上方绘制文字标签
                    cv2.putText(
                        display,
                        label,
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_color,
                        1
                    )

                    # 绘制左右手腕关键点，黄色
                    for side, idx in (("L", LEFT_WRIST), ("R", RIGHT_WRIST)):
                        if valid_kpt(kpts, idx):
                            u, v = int(kpts[idx][0]), int(kpts[idx][1])
                            cv2.circle(display, (u, v), 4, (0, 255, 255), -1)

                    # 绘制左右肩膀关键点，紫色
                    for side, idx in (("L", LEFT_SHOULDER), ("R", RIGHT_SHOULDER)):
                        if valid_kpt(kpts, idx):
                            u, v = int(kpts[idx][0]), int(kpts[idx][1])
                            cv2.circle(display, (u, v), 4, (255, 0, 255), -1)

            # 重点：根据 show_mask 决定显示普通检测画面，还是同时显示红色掩膜
            if show_mask:
                mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Teammate Detection Test", np.hstack((display, mask_bgr)))
            else:
                cv2.imshow("Teammate Detection Test", display)

            # 重点：读取键盘输入
            key = cv2.waitKey(1) & 0xFF

            # 按 q 退出程序
            if key == ord('q'):
                break

            # 按 m 切换是否显示红色掩膜
            elif key == ord('m'):
                show_mask = not show_mask

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print("\n程序被中断")

    finally:
        # 重点：无论正常退出还是异常退出，都要释放窗口和相机资源
        cv2.destroyAllWindows()
        pipeline.stop()
        print("程序结束")


# Python 程序入口
if __name__ == "__main__":
    main()