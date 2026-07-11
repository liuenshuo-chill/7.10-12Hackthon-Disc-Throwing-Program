#!/usr/bin/env python3
"""
Intel RealSense D405 - 红色飞盘检测（正面/侧面识别）
优化点：
  1. 双边滤波 + CLAHE 提升光照鲁棒性
  2. 跨帧追踪器：短暂丢失时保持上一帧结果（最多 MAX_LOST_FRAMES 帧）
  3. 圆心坐标指数平滑，消除抖动
  4. 深度取邻域中值，避免单点噪声
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# ── 相机配置 ──────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FPS = 30

# ── D405 深度范围（毫米）─────────────────────────────────
MIN_DEPTH_MM = 70
MAX_DEPTH_MM = 1000

# ── 红色 HSV 范围（可根据实际光照微调）────────────────────
RED_HSV_LOWER1 = np.array([0, 60, 40])
RED_HSV_UPPER1 = np.array([10, 255, 255])
RED_HSV_LOWER2 = np.array([170, 60, 40])
RED_HSV_UPPER2 = np.array([180, 255, 255])

# ── 面积/尺寸限制（像素）──────────────────────────────────
MIN_CONTOUR_AREA = 300     # 过滤掉过小的噪声轮廓
MIN_AXIS_LEN     = 15      # 拟合椭圆短轴最小长度（像素）
MAX_AXIS_LEN     = 400     # 拟合椭圆长轴最大长度（像素）

# ── 正面/侧面判别阈值（长轴/短轴比例）────────────────────
FRONT_ASPECT_MAX = 1.35   # 低于此值判定为"正面"
SIDE_ASPECT_MIN  = 1.8    # 高于此值判定为"侧面"

# ── 追踪器参数 ────────────────────────────────────────────
MAX_LOST_FRAMES = 8    # 丢失超过此帧数才真正移除目标
SMOOTH_ALPHA    = 0.4  # 指数平滑系数（越小越平滑，越大越灵敏）
MATCH_DIST_PX   = 60   # 两帧间同一目标最大位移（像素）


# ─────────────────────────────────────────────────────────
class CircleTracker:
    """跨帧追踪器（支持正面/侧面两种视角）"""

    def __init__(self):
        # 每条轨迹: {'cx','cy','view','angle_deg','major','minor','lost','x3d','y3d','z3d'}
        self.tracks = []
        self._next_id = 0

    def update(self, detections):
        """
        detections: list of dict，每个元素为
            {'cx','cy','view','angle_deg','major','minor'}
        """
        unmatched_det = list(range(len(detections)))

        for t in self.tracks:
            best_i, best_d = None, MATCH_DIST_PX
            for i in unmatched_det:
                d_cx, d_cy = detections[i]['cx'], detections[i]['cy']
                d = np.hypot(d_cx - t['cx'], d_cy - t['cy'])
                if d < best_d:
                    best_d, best_i = d, i

            if best_i is not None:
                det = detections[best_i]
                t['cx'] = int(SMOOTH_ALPHA * det['cx'] + (1 - SMOOTH_ALPHA) * t['cx'])
                t['cy'] = int(SMOOTH_ALPHA * det['cy'] + (1 - SMOOTH_ALPHA) * t['cy'])
                t['major'] = det['major']
                t['minor'] = det['minor']
                t['view'] = det['view']
                t['angle_deg'] = det['angle_deg']
                t['lost'] = 0
                unmatched_det.remove(best_i)
            else:
                t['lost'] += 1

        self.tracks = [t for t in self.tracks if t['lost'] <= MAX_LOST_FRAMES]

        for i in unmatched_det:
            det = detections[i]
            self.tracks.append({
                'id': self._next_id,
                'cx': det['cx'], 'cy': det['cy'],
                'view': det['view'], 'angle_deg': det['angle_deg'],
                'major': det['major'], 'minor': det['minor'],
                'lost': 0, 'x3d': 0.0, 'y3d': 0.0, 'z3d': 0.0
            })
            self._next_id += 1

        return self.tracks


# ─────────────────────────────────────────────────────────
def preprocess(color_image):
    """双边滤波 + V 通道 CLAHE，提升光照鲁棒性"""
    blurred = cv2.bilateralFilter(color_image, d=9, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return hsv


def detect_red_disc(color_image):
    """
    检测红色飞盘，识别正面(front)/侧面(side)
    返回：
        detections: list of dict，每个元素为
            {'cx','cy','view','angle_deg','major','minor'}
        mask: 二值化掩膜（调试用）
    """
    hsv = preprocess(color_image)
    mask1 = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # fitEllipse 至少需要 5 个轮廓点
        if len(cnt) < 5:
            continue

        (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(cnt)
        major, minor = max(axis1, axis2), min(axis1, axis2)

        if minor < MIN_AXIS_LEN or major > MAX_AXIS_LEN:
            continue
        if minor < 1e-3:
            continue

        aspect = major / minor

        # angle对应axis1方向；若axis2才是长轴，方向需要旋转90°修正
        major_angle = angle if axis1 >= axis2 else (angle + 90.0) % 180.0

        if aspect <= FRONT_ASPECT_MAX:
            view = 'front'
            angle_deg = None
        elif aspect >= SIDE_ASPECT_MIN:
            view = 'side'
            angle_deg = major_angle % 180.0
        else:
            # 过渡地带，判定不确定，丢弃避免正面/侧面来回跳变
            continue

        detections.append({
            'cx': int(cx), 'cy': int(cy),
            'view': view, 'angle_deg': angle_deg,
            'major': major, 'minor': minor
        })

    return detections, mask


def get_median_depth(depth_frame, cx, cy, depth_scale, min_units, max_units, patch=5):
    """在 (cx,cy) 邻域取中值深度，用 depth_scale 转换为米"""
    h, w = HEIGHT, WIDTH
    x0, x1 = max(0, cx - patch), min(w, cx + patch + 1)
    y0, y1 = max(0, cy - patch), min(h, cy + patch + 1)
    depth_arr = np.asanyarray(depth_frame.get_data())[y0:y1, x0:x1].astype(np.float32)
    valid = depth_arr[(depth_arr >= min_units) & (depth_arr <= max_units)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * depth_scale  # units -> m


def pixel_to_3d(u, v, depth_m, intrinsic):
    """使用 pyrealsense2 官方反投影接口，精度更高"""
    point = rs.rs2_deproject_pixel_to_point(intrinsic, [float(u), float(v)], depth_m)
    return point[0], point[1], point[2]


def draw_results(display, tracks, nearest=None):
    """正面(front)：绿色圆；侧面(side)：黄色线段标出长轴方向"""
    active = sum(1 for t in tracks if t['lost'] == 0)

    for t in tracks:
        if t['lost'] > 0:
            continue
        cx, cy = t['cx'], t['cy']
        is_nearest = nearest and t['id'] == nearest['id']
        base_color = (0, 255, 0) if is_nearest else (120, 120, 120)

        if t['view'] == 'front':
            r = int(t['major'] / 2)
            cv2.circle(display, (cx, cy), r, base_color, 2 if is_nearest else 1)
            label = "FRONT"
        else:
            angle = t['angle_deg'] if t['angle_deg'] is not None else 0.0
            rad = np.deg2rad(angle)
            half_len = t['major'] / 2
            dx, dy = np.cos(rad) * half_len, np.sin(rad) * half_len
            p1 = (int(cx - dx), int(cy - dy))
            p2 = (int(cx + dx), int(cy + dy))
            cv2.line(display, p1, p2, (0, 255, 255) if is_nearest else base_color,
                     3 if is_nearest else 1)
            label = f"SIDE {angle:.1f}deg"

        if is_nearest:
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            x3d, y3d, z3d = t['x3d'], t['y3d'], t['z3d']
            depth_label = (f"X:{x3d:.3f} Y:{y3d:.3f} Z:{z3d:.3f}m"
                           if z3d > 0 else "depth invalid")
            # 这两行cv2.putText就是屏幕显示"FRONT"或"SIDE xx.xdeg"文字的地方
            cv2.putText(display, label, (cx - 60, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, depth_label, (cx - 60, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(display, f"Red discs: {active}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)


def main():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("启动RealSense D405相机...")
    profile = pipeline.start(config)

    # 读取实际 depth scale（单位：米/unit），避免硬编码单位假设
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")

    # 将 MIN/MAX 深度转换为 depth frame 原始单位
    min_depth_units = MIN_DEPTH_MM / 1000.0 / depth_scale
    max_depth_units = MAX_DEPTH_MM / 1000.0 / depth_scale

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsic = color_profile.get_intrinsics()
    print(f"相机内参: fx={intrinsic.fx:.2f} fy={intrinsic.fy:.2f} "
          f"cx={intrinsic.ppx:.2f} cy={intrinsic.ppy:.2f}")

    align     = rs.align(rs.stream.color)
    tracker   = CircleTracker()
    show_mask = False

    print("按 'q' 退出  |  'm' 切换掩膜")

    try:
        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame    = aligned_frames.get_depth_frame()
            color_frame    = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 检测 + 追踪
            detections, mask = detect_red_disc(color_image)       
            tracks = tracker.update(detections)

            # 更新活跃轨迹的3D坐标
            for t in tracks:
                if t['lost'] == 0:
                    depth_m = get_median_depth(depth_frame, t['cx'], t['cy'],
                                               depth_scale, min_depth_units, max_depth_units)
                    if depth_m > 0:
                        t['x3d'], t['y3d'], t['z3d'] = pixel_to_3d(
                            t['cx'], t['cy'], depth_m, intrinsic)

            # 找出最近的活跃轨迹（Z最小且有效）
            active = [t for t in tracks if t['lost'] == 0 and t['z3d'] > 0]
            nearest = min(active, key=lambda t: t['z3d']) if active else None

            if nearest:
                if nearest['view'] == 'side':
                    print(f"  [侧面] 中心({nearest['cx']},{nearest['cy']}) "
                        f"角度={nearest['angle_deg']:.1f}° Z={nearest['z3d']:.4f}m")
                else:
                    print(f"  [正面] 中心({nearest['cx']},{nearest['cy']}) "
                        f"Z={nearest['z3d']:.4f}m")

            # 绘制（只框选最近目标）
            display = color_image.copy()
            draw_results(display, tracks, nearest)

            if show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Red Disc Detection', np.hstack((display, mask_bgr)))
            else:
                cv2.imshow('Red Disc Detection', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask

    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == "__main__":
    main()
