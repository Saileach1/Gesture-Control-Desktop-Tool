"""
手势控制桌面工具
使用OpenCV和MediaPipe实现的手势识别和桌面控制功能
"""
import cv2
import pyautogui
import math
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np


class GestureController:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        # 使用MediaPipe的新API - HandLandmarker
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # 创建 HandLandmarker
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # 获取绘制工具
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.hand_landmarks_connections = mp.tasks.vision.HandLandmarksConnections

        # 鼠标控制状态
        self.is_controlling = False
        self.mouse_sensitivity = 0.15  # 大幅降低鼠标灵敏度
        self.click_threshold = 30  # 像素阈值，用于判断点击

        # 存储上一帧手指位置
        self.prev_x, self.prev_y = None, None

        # 鼠标平滑控制
        self.mouse_x_history = []
        self.mouse_y_history = []
        self.history_length = 5  # 历史位置数量，用于平滑
        self.min_movement_threshold = 2  # 最小移动阈值，防止微小抖动
        self.max_movement_threshold = 100  # 最大移动阈值，增大以覆盖屏幕边缘

        # 控制参数
        self.gesture_commands = {
            'open_palm': '单击',
            'point_up': '鼠标移动',
            'victory': '双击',
            'thumb_up': '滚动上',
            'thumb_down': '滚动下',
            'ok_gesture': '右键单击'
        }

        # 设置PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01  # 减少操作间隔，提高流畅度

        # 外部控制开关回调函数（可选）
        self.enable_control_callback = None

    def get_available_cameras(self):
        """获取系统中可用的摄像头列表"""
        available_cameras = []
        for i in range(10):  # 检查前10个摄像头设备
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras

    def validate_hand_landmarks(self, landmarks):
        """验证手部关键点是否有效，避免误识别非手部物体"""
        # 检查是否有21个关键点
        if len(landmarks) != 21:
            return False

        # 检查所有关键点的坐标有效性
        for landmark in landmarks:
            # 检查坐标是否在有效范围内 (0-1)
            if landmark.x < 0 or landmark.x > 1 or landmark.y < 0 or landmark.y > 1:
                return False

        # 检查手掌区域是否合理（掌心到指尖的距离）
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # 计算手掌大小
        palm_size = (
            self.calculate_distance((wrist.x, wrist.y), (index_tip.x, index_tip.y)) +
            self.calculate_distance((wrist.x, wrist.y), (middle_tip.x, middle_tip.y))
        ) / 2

        # 手掌大小应该在合理范围内（0.1到0.5之间）
        if palm_size < 0.1 or palm_size > 0.5:
            return False

        return True

    def process_frame_legacy(self, frame):
        """使用传统轮廓检测方法处理单帧图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)
        
        # 二值化
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 复制帧用于显示
        display_frame = frame.copy()
        
        gesture = None
        # 找到最大的轮廓（假设是手）
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand_contour) > 10000:  # 过滤小的轮廓
                # 绘制手部轮廓
                cv2.drawContours(display_frame, [hand_contour], -1, (0, 255, 0), 2)
                
                # 识别手势
                gesture = self.get_gesture_legacy(hand_contour, frame)
                
                # 执行手势对应的操作
                if gesture and self.is_controlling:
                    self.execute_gesture_command_legacy(gesture, frame)
        
        return display_frame, gesture

    def get_gesture_legacy(self, hand_contour, frame):
        """通过图像处理来识别手势（传统方法）"""
        if hand_contour is None:
            return None
        
        finger_count, finger_tips = self.detect_fingers_legacy(hand_contour)
        
        # 根据手指数量判断手势
        if finger_count == 0:
            # 握拳 - 不再识别为任何手势
            return None
        elif finger_count == 1:
            # 尝试区分食指上指和下指
            # 这里需要更复杂的逻辑来确定是食指上指还是其他单指手势
            return 'point_up'  # 默认为食指上指
        elif finger_count == 2:
            return 'victory'  # 胜利手势
        elif finger_count == 4 or finger_count == 5:
            return 'open_palm'  # 张开手掌
        else:
            # 更复杂的识别逻辑
            # 检查是否是OK手势（拇指和食指形成圆圈）
            if len(finger_tips) >= 2:
                # 检查是否有两个相邻的手指尖距离很近（形成OK手势）
                for i in range(len(finger_tips)):
                    for j in range(i+1, len(finger_tips)):
                        dist = self.calculate_distance(finger_tips[i], finger_tips[j])
                        if dist < 50:  # 如果两个指尖距离很近，认为是OK手势
                            return 'ok_gesture'
        
        return None

    def detect_fingers_legacy(self, hand_contour):
        """检测手指数量（传统方法）"""
        if hand_contour is None:
            return 0, []
        
        # 计算轮廓的凸包
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        
        if len(hull) < 3:
            return 0, []
        
        # 找到凸包的缺陷
        defects = cv2.convexityDefects(hand_contour, hull)
        
        if defects is None:
            return 0, []
        
        # 统计手指数量
        finger_count = 0
        finger_tips = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            
            # 计算三角形的三边
            a = self.calculate_distance(far, end)
            b = self.calculate_distance(far, start)
            c = self.calculate_distance(start, end)
            
            # 使用余弦定理计算角度
            angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b)) * 180 / math.pi
            
            # 角度小于90度，且深度大于一定阈值，则认为是手指凹陷
            if angle <= 90 and d > 10000:
                finger_count += 1
                finger_tips.append(start)  # 手指尖位置
        
        # 加上拇指
        return finger_count + 1, finger_tips

    def execute_gesture_command_legacy(self, gesture, frame):
        """执行手势命令（传统方法）"""
        if gesture == 'point_up':  # 食指上指 - 鼠标移动和单击
            # 使用轮廓的重心作为鼠标位置
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (35, 35), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hand_contour = max(contours, key=cv2.contourArea)
                if hand_contour is not None:
                    M = cv2.moments(hand_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 将坐标映射到屏幕尺寸
                        screen_width, screen_height = pyautogui.size()
                        x = int((cx / frame.shape[1]) * screen_width)
                        y = int((cy / frame.shape[0]) * screen_height)
                        
                        # 限制在屏幕范围内
                        x = max(0, min(x, screen_width - 1))
                        y = max(0, min(y, screen_height - 1))
                        
                        # 移动鼠标（进一步降低灵敏度，添加平滑移动）
                        current_x, current_y = pyautogui.position()
                        smooth_x = int(current_x * 0.8 + x * 0.2)  # 进一步降低灵敏度，80%当前鼠标位置 + 20%手势检测位置
                        smooth_y = int(current_y * 0.8 + y * 0.2)
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0.3)  # 增加移动时间，使更平滑
                        
                        # 模拟点击
                        pyautogui.click()
        
        elif gesture == 'point_down':  # 食指下指 - 右键单击
            # 使用轮廓的重心作为鼠标位置
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (35, 35), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hand_contour = max(contours, key=cv2.contourArea)
                if hand_contour is not None:
                    M = cv2.moments(hand_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 将坐标映射到屏幕尺寸
                        screen_width, screen_height = pyautogui.size()
                        x = int((cx / frame.shape[1]) * screen_width)
                        y = int((cy / frame.shape[0]) * screen_height)
                        
                        # 限制在屏幕范围内
                        x = max(0, min(x, screen_width - 1))
                        y = max(0, min(y, screen_height - 1))
                        
                        # 移动鼠标（进一步降低灵敏度，添加平滑移动）
                        current_x, current_y = pyautogui.position()
                        smooth_x = int(current_x * 0.8 + x * 0.2)  # 进一步降低灵敏度，80%当前鼠标位置 + 20%手势检测位置
                        smooth_y = int(current_y * 0.8 + y * 0.2)
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0.3)  # 增加移动时间，使更平滑
                        
                        # 模拟右键单击
                        pyautogui.rightClick()
            
        elif gesture == 'victory':  # 胜利手势 - 双击
            # 延迟双击，使操作更平滑
            pyautogui.doubleClick(interval=0.3)  # 增加双击间隔时间
            
        elif gesture == 'thumb_up':  # 竖大拇指向上 - 向上滚动
            pyautogui.scroll(10)  # 滚动量
            
        elif gesture == 'thumb_down':  # 竖大拇指向下 - 向下滚动
            pyautogui.scroll(-10)  # 滚动量
            
        elif gesture == 'ok_gesture':  # OK手势 - 右键单击
            pyautogui.rightClick()

    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_gesture_from_landmarks(self, landmarks, image_width, image_height):
        """根据landmarks识别手势 - 使用关节点精确判断"""
        if len(landmarks) < 21:
            return None

        # 定义手指关键点索引
        # 拇指: 0(腕)-1(拇指根部)-2(掌指)-3(近节)-4(指尖)
        # 食指: 0(腕)-5(掌指)-6(近节)-7(中节)-8(指尖)
        # 中指: 0(腕)-9(掌指)-10(近节)-11(中节)-12(指尖)
        # 无名指: 0(腕)-13(掌指)-14(近节)-15(中节)-16(指尖)
        # 小指: 0(腕)-17(掌指)-18(近节)-19(中节)-20(指尖)

        # 计算手指是否伸直（通过比较关节点的Y坐标）
        def is_finger_extended(tip_idx, pip_idx, mcp_idx):
            """判断手指是否伸直 - 使用更宽松的条件"""
            tip_y = landmarks[tip_idx].y
            pip_y = landmarks[pip_idx].y
            mcp_y = landmarks[mcp_idx].y

            # 方法1: 比较Y坐标
            # 如果指尖的Y坐标比PIP（近节）小（在上方），或者差距很小，则认为伸直
            # 使用0.02的容差，避免微小误差影响判断
            is_extended_y = tip_y <= pip_y + 0.02

            # 方法2: 计算手指弯曲程度
            # 计算指尖到掌指关节的距离
            tip_mcp_distance = math.sqrt(
                (landmarks[tip_idx].x - landmarks[mcp_idx].x)**2 +
                (landmarks[tip_idx].y - landmarks[mcp_idx].y)**2
            )

            # 如果指尖到MCP的距离大于某个阈值，认为手指伸直
            is_extended_distance = tip_mcp_distance > 0.15

            # 只要满足任一条件，就认为手指伸直
            return is_extended_y or is_extended_distance

        # 判断拇指是否伸出（平衡版）
        def is_thumb_extended():
            """判断拇指是否伸出 - 平衡版"""
            # 方法1: 比较拇指指尖和拇指IP关节的位置
            thumb_tip_x = landmarks[4].x
            thumb_tip_y = landmarks[4].y
            thumb_ip_x = landmarks[3].x
            thumb_ip_y = landmarks[3].y
            thumb_mcp_x = landmarks[2].x
            thumb_mcp_y = landmarks[2].y

            # 计算拇指指尖到IP关节的距离
            distance_tip_ip = math.sqrt((thumb_tip_x - thumb_ip_x)**2 + (thumb_tip_y - thumb_ip_y)**2)
            # 计算拇指IP关节到MCP关节的距离
            distance_ip_mcp = math.sqrt((thumb_ip_x - thumb_mcp_x)**2 + (thumb_ip_y - thumb_mcp_y)**2)

            # 如果指尖到IP的距离大于IP到MCP的距离，说明拇指伸出
            # 使用0.7的比例，平衡识别能力
            method1 = distance_tip_ip > distance_ip_mcp * 0.7

            # 方法2: 检查拇指指尖是否远离手掌中心
            # 计算手掌中心（使用食指、中指、无名指、小指的MCP关节）
            palm_center_x = (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4
            palm_center_y = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4

            # 计算拇指指尖到手掌中心的距离
            thumb_to_palm = math.sqrt((thumb_tip_x - palm_center_x)**2 + (thumb_tip_y - palm_center_y)**2)

            # 如果拇指指尖距离手掌中心较远，认为拇指伸出
            # 降低阈值到0.2
            method2 = thumb_to_palm > 0.2

            # 只需要满足任一方法就认为拇指伸出（使用or而不是and）
            return method1 or method2

        # 判断各手指状态
        index_extended = is_finger_extended(8, 6, 5)   # 食指
        middle_extended = is_finger_extended(12, 10, 9)  # 中指
        ring_extended = is_finger_extended(16, 14, 13)   # 无名指
        pinky_extended = is_finger_extended(20, 18, 17)   # 小指
        thumb_extended = is_thumb_extended()

        # 计算拇指和食指指尖距离（用于OK手势）
        thumb_tip = (landmarks[4].x * image_width, landmarks[4].y * image_height)
        index_tip = (landmarks[8].x * image_width, landmarks[8].y * image_height)
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)

        # 识别不同手势 - 调整优先级

        # 1. OK手势 - 拇指和食指形成圆圈（距离很近）
        if thumb_index_dist < 50 and middle_extended and ring_extended and pinky_extended:
            return 'ok_gesture'

        # 2. 食指上指 - 只有食指伸直（鼠标移动）
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'point_up'

        # 3. 胜利手势 - 食指和中指伸直
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'victory'

        # 4. 竖大拇指 - 通过拇指与其他手指的夹角判断
        # 条件：四指弯曲，且拇指与食指的夹角较大
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # 四指已经弯曲，现在检查拇指与食指的夹角
            
            # 获取拇指指尖和食指MCP关节的位置
            thumb_tip_x = landmarks[4].x
            thumb_tip_y = landmarks[4].y
            index_mcp_x = landmarks[5].x
            index_mcp_y = landmarks[5].y
            
            # 计算拇指向量（从食指MCP到拇指指尖）
            thumb_vector_x = thumb_tip_x - index_mcp_x
            thumb_vector_y = thumb_tip_y - index_mcp_y
            
            # 获取食指指尖位置
            index_tip_x = landmarks[8].x
            index_tip_y = landmarks[8].y
            
            # 计算食指向量（从食指MCP到食指指尖）
            index_vector_x = index_tip_x - index_mcp_x
            index_vector_y = index_tip_y - index_mcp_y
            
            # 计算两个向量的夹角
            # 使用点积公式：cos(angle) = (a·b) / (|a|*|b|)
            dot_product = thumb_vector_x * index_vector_x + thumb_vector_y * index_vector_y
            thumb_magnitude = math.sqrt(thumb_vector_x**2 + thumb_vector_y**2)
            index_magnitude = math.sqrt(index_vector_x**2 + index_vector_y**2)
            
            if thumb_magnitude > 0 and index_magnitude > 0:
                cos_angle = dot_product / (thumb_magnitude * index_magnitude)
                # 限制cos值在[-1, 1]范围内，避免数值误差
                cos_angle = max(-1, min(1, cos_angle))
                angle_between = math.degrees(math.acos(cos_angle))
                
                # 如果拇指与食指的夹角大于 60 度，认为拇指是竖起的
                if angle_between > 60:
                    # 判断拇指方向（向上或向下）
                    # 比较拇指指尖和食指MCP的Y坐标
                    if thumb_tip_y < index_mcp_y - 0.05:
                        return 'thumb_up'
                    elif thumb_tip_y > index_mcp_y + 0.05:
                        return 'thumb_down'

        

        # 6. 张开手掌 - 四指伸直即可（拇指可以自由）
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return 'open_palm'

        return None

    def process_frame(self, frame):
        """处理单帧图像 - 使用 MediaPipe HandLandmarker"""
        # 图像增强处理
        # 1. 增加对比度
        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        # 2. 转换为RGB格式
        image_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)

        # 将图像转换为 MediaPipe Image 对象
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 检测手部关键点
        detection_result = self.hand_landmarker.detect(mp_image)

        gesture = None
        display_frame = frame_enhanced.copy()

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # 转换为标准的 landmark 格式
                landmarks = hand_landmarks

                # 验证手部关键点是否有效
                if not self.validate_hand_landmarks(landmarks):
                    continue

                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    landmarks,
                    self.hand_landmarks_connections.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 识别手势（基于关节点）
                gesture = self.get_gesture_from_landmarks(landmarks, frame.shape[1], frame.shape[0])

                # 检查是否启用控制
                enable_control = True
                if self.enable_control_callback:
                    enable_control = self.enable_control_callback()

                # 执行手势对应的操作（只有在启用控制且控制状态为激活时）
                if gesture and self.is_controlling and enable_control:
                    self.execute_gesture_command(gesture, landmarks, frame.shape[1], frame.shape[0])

        return display_frame, gesture

    def smooth_mouse_move(self, target_x, target_y):
        """平滑鼠标移动，使用历史位置平均和阈值过滤"""
        # 获取当前鼠标位置
        current_x, current_y = pyautogui.position()

        # 计算移动距离
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx**2 + dy**2)

        # 如果移动距离太小，忽略（防抖动）
        if distance < self.min_movement_threshold:
            return

        # 如果移动距离太大，限制最大移动（防跳跃）
        if distance > self.max_movement_threshold:
            scale = self.max_movement_threshold / distance
            target_x = current_x + dx * scale
            target_y = current_y + dy * scale

        # 添加到历史位置
        self.mouse_x_history.append(target_x)
        self.mouse_y_history.append(target_y)

        # 保持历史位置数量
        if len(self.mouse_x_history) > self.history_length:
            self.mouse_x_history.pop(0)
            self.mouse_y_history.pop(0)

        # 计算平均位置
        avg_x = int(sum(self.mouse_x_history) / len(self.mouse_x_history))
        avg_y = int(sum(self.mouse_y_history) / len(self.mouse_y_history))

        # 限制在屏幕范围内
        screen_width, screen_height = pyautogui.size()
        avg_x = max(0, min(avg_x, screen_width - 1))
        avg_y = max(0, min(avg_y, screen_height - 1))

        # 直接移动到目标位置，不使用 duration 参数（避免卡顿）
        pyautogui.moveTo(avg_x, avg_y)

    def execute_gesture_command(self, gesture, landmarks, image_width, image_height):
        """执行手势命令"""
        if gesture == 'point_up':  # 食指上指 - 鼠标移动（不点击）
            # 获取食指指尖位置
            index_tip = landmarks[8]
            x = int(index_tip.x * pyautogui.size().width)
            y = int(index_tip.y * pyautogui.size().height)

            # 使用平滑鼠标移动
            self.smooth_mouse_move(x, y)

        elif gesture == 'open_palm':  # 张开手掌 - 单击
            pyautogui.click()

        elif gesture == 'victory':  # 胜利手势 - 双击
            # 延迟双击，使操作更平滑
            pyautogui.doubleClick(interval=0.5)  # 增加双击间隔时间，进一步平滑操作

        elif gesture == 'thumb_up':  # 竖大拇指向上 - 向上滚动
            pyautogui.scroll(10)  # 滚动量

        elif gesture == 'thumb_down':  # 竖大拇指向下 - 向下滚动
            pyautogui.scroll(-10)  # 滚动量

        elif gesture == 'ok_gesture':  # OK手势 - 右键单击
            pyautogui.rightClick()

    def execute_gesture_command_for_legacy(self, gesture, frame):
        """为传统方法执行手势命令"""
        if gesture == 'point_up':  # 食指上指 - 鼠标移动和单击
            # 使用轮廓的重心作为鼠标位置
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (35, 35), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hand_contour = max(contours, key=cv2.contourArea)
                if hand_contour is not None:
                    M = cv2.moments(hand_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 将坐标映射到屏幕尺寸
                        screen_width, screen_height = pyautogui.size()
                        x = int((cx / frame.shape[1]) * screen_width)
                        y = int((cy / frame.shape[0]) * screen_height)
                        
                        # 限制在屏幕范围内
                        x = max(0, min(x, screen_width - 1))
                        y = max(0, min(y, screen_height - 1))
                        
                        # 移动鼠标
                        pyautogui.moveTo(x, y, duration=0.1)
                        
                        # 模拟点击
                        pyautogui.click()
            
        elif gesture == 'victory':  # 胜利手势 - 双击
            # 延迟双击，使操作更平滑
            pyautogui.doubleClick(interval=0.5)  # 增加双击间隔时间，进一步平滑操作
            
        elif gesture == 'thumb_up':  # 竖大拇指向上 - 向上滚动
            pyautogui.scroll(10)  # 滚动量
            
        elif gesture == 'thumb_down':  # 竖大拇指向下 - 向下滚动
            pyautogui.scroll(-10)  # 滚动量
            
        elif gesture == 'ok_gesture':  # OK手势 - 右键单击
            pyautogui.rightClick()

    def start_control(self):
        """开始手势控制"""
        self.is_controlling = True
        
    def stop_control(self):
        """停止手势控制"""
        self.is_controlling = False

    def run_camera(self):
        """运行摄像头并进行手势识别"""
        cap = cv2.VideoCapture(self.camera_id)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 翻转图像
            frame = cv2.flip(frame, 1)

            # 检查外部控制开关
            enable_control = True
            if self.enable_control_callback:
                enable_control = self.enable_control_callback()

            # 处理图像
            processed_frame, gesture = self.process_frame(frame)

            # 显示当前手势
            if gesture:
                cv2.putText(processed_frame, f'Gesture: {gesture}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示控制状态
            status_text = "ACTIVE" if self.is_controlling and enable_control else "PAUSED"
            color = (0, 255, 0) if self.is_controlling and enable_control else (0, 0, 255)
            cv2.putText(processed_frame, f'Status: {status_text}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 显示操作开关状态
            action_status = "ENABLED" if enable_control else "DISABLED"
            action_color = (0, 255, 0) if enable_control else (255, 165, 0)
            cv2.putText(processed_frame, f'Action: {action_status}', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, action_color, 2)

            # 在图像上显示提示信息
            cv2.putText(processed_frame, "Press SPACE to toggle control, 'q' to quit",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Gesture Control', processed_frame)

            # 按'q'退出，按空格键切换控制状态
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.is_controlling = not self.is_controlling
                print(f"Control toggled: {'ACTIVE' if self.is_controlling else 'PAUSED'}")  # 调试输出

        cap.release()
        cv2.destroyAllWindows()


class GestureControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("手势控制桌面工具")
        self.root.geometry("600x750")
        self.root.configure(bg='#f0f0f0')
        
        # 创建手势控制器实例
        self.controller = GestureController()

        # 获取可用摄像头
        self.available_cameras = self.controller.get_available_cameras()
        if not self.available_cameras:
            self.available_cameras = [0]  # 默认使用摄像头0

        # 设置界面
        self.setup_ui()

        # 控制线程
        self.control_thread = None
        self.running = False

        # 启用控制开关（默认开启）
        self.enable_control = True

    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(self.root, text="手势控制桌面工具",
                              font=("Arial", 20, "bold"),
                              bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # 摄像头选择框架
        camera_frame = tk.LabelFrame(self.root, text="摄像头设置", font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#34495e', padx=10, pady=10)
        camera_frame.pack(fill="x", padx=20, pady=10)
        
        camera_select_frame = tk.Frame(camera_frame, bg='#f0f0f0')
        camera_select_frame.pack(fill="x")
        
        tk.Label(camera_select_frame, text="选择摄像头:", font=("Arial", 11),
                bg='#f0f0f0', fg='#34495e').pack(side="left", padx=5)
        
        # 创建摄像头下拉选择框
        camera_options = [f"摄像头 {i}" for i in self.available_cameras]
        self.camera_var = tk.StringVar(value=f"摄像头 {self.available_cameras[0]}")
        self.camera_combo = ttk.Combobox(camera_select_frame, textvariable=self.camera_var,
                                        values=camera_options, state="readonly", width=10)
        self.camera_combo.pack(side="left", padx=5)
        
        # 摄像头刷新按钮
        refresh_button = tk.Button(camera_select_frame, text="刷新", command=self.refresh_cameras,
                                  font=("Arial", 10), bg='#3498db', fg='white', width=8)
        refresh_button.pack(side="left", padx=5)
        
        # 功能说明框架
        info_frame = tk.LabelFrame(self.root, text="功能说明", font=("Arial", 12, "bold"),
                                  bg='#f0f0f0', fg='#34495e', padx=10, pady=10)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        info_text = """
        • 张开手掌 - 单击
        • 食指上指 - 鼠标移动
        • 胜利手势 (V) - 双击
        • 竖大拇指向上 - 向上滚动
        • 竖大拇指向下 - 向下滚动
        • OK手势 - 右键单击
        """
        info_label = tk.Label(info_frame, text=info_text, justify="left",
                             bg='#f0f0f0', fg='#34495e', font=("Arial", 10))
        info_label.pack()
        
        # 控制面板
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill="x", padx=20, pady=20)

        # 左侧按钮组
        left_buttons = tk.Frame(control_frame, bg='#f0f0f0')
        left_buttons.pack(side="left", padx=10)

        # 开始/停止按钮
        self.start_button = tk.Button(left_buttons, text="开始控制",
                                     command=self.toggle_control,
                                     font=("Arial", 12, "bold"),
                                     bg='#27ae60', fg='white',
                                     width=12, height=2)
        self.start_button.pack(side="left", padx=5)

        # 启用控制开关按钮
        self.enable_control_button = tk.Button(left_buttons, text="启用操作: 开",
                                             command=self.toggle_enable_control,
                                             font=("Arial", 12, "bold"),
                                             bg='#3498db', fg='white',
                                             width=12, height=2)
        self.enable_control_button.pack(side="left", padx=5)

        # 退出按钮
        exit_button = tk.Button(control_frame, text="退出",
                               command=self.exit_app,
                               font=("Arial", 12, "bold"),
                               bg='#e74c3c', fg='white',
                               width=12, height=2)
        exit_button.pack(side="right", padx=10)
        
        # 状态显示
        self.status_label = tk.Label(self.root, text="状态: 未启动", 
                                    font=("Arial", 14), 
                                    bg='#f0f0f0', fg='#34495e')
        self.status_label.pack(pady=10)
        
        # 手势说明框架
        gesture_frame = tk.LabelFrame(self.root, text="手势对照表", 
                                     font=("Arial", 12, "bold"),
                                     bg='#f0f0f0', fg='#34495e', 
                                     padx=10, pady=10)
        gesture_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 创建手势列表
        gestures = [
            ("张开手掌", "单击"),
            ("食指上指", "鼠标移动"),
            ("胜利手势 (V)", "双击"),
            ("竖大拇指向上", "向上滚动"),
            ("竖大拇指向下", "向下滚动"),
            ("OK手势", "右键单击")
        ]
        
        for i, (gesture, description) in enumerate(gestures):
            row_frame = tk.Frame(gesture_frame, bg='#f0f0f0')
            row_frame.pack(fill="x", pady=5)
            
            gesture_label = tk.Label(row_frame, text=f"• {gesture}", 
                                   font=("Arial", 11, "bold"),
                                   bg='#f0f0f0', fg='#2c3e50', width=15, anchor="w")
            gesture_label.pack(side="left")
            
            desc_label = tk.Label(row_frame, text=description, 
                                font=("Arial", 11),
                                bg='#f0f0f0', fg='#7f8c8d')
            desc_label.pack(side="left", padx=20)
    
    def refresh_cameras(self):
        """刷新可用摄像头列表"""
        self.available_cameras = self.controller.get_available_cameras()
        if not self.available_cameras:
            self.available_cameras = [0]
        
        camera_options = [f"摄像头 {i}" for i in self.available_cameras]
        self.camera_combo['values'] = camera_options
        if camera_options:
            self.camera_var.set(camera_options[0])
    
    def toggle_control(self):
        """切换控制状态"""
        if not self.running:
            # 获取选择的摄像头编号
            camera_selection = self.camera_var.get()
            camera_id = int(camera_selection.split()[-1])
            self.controller.camera_id = camera_id

            # 设置启用控制的回调函数
            self.controller.enable_control_callback = lambda: self.enable_control

            self.running = True
            self.start_button.config(text="停止控制", bg='#e74c3c')
            self.status_label.config(text="状态: 运行中")
            # 在启动摄像头前确保控制器状态为True
            self.controller.is_controlling = True
            self.control_thread = threading.Thread(target=self.run_controller)
            self.control_thread.daemon = True
            self.control_thread.start()
        else:
            self.running = False
            self.start_button.config(text="开始控制", bg='#27ae60')
            self.status_label.config(text="状态: 已停止")

    def toggle_enable_control(self):
        """切换启用控制状态"""
        self.enable_control = not self.enable_control
        if self.enable_control:
            self.enable_control_button.config(text="启用操作: 开", bg='#3498db')
        else:
            self.enable_control_button.config(text="启用操作: 关", bg='#95a5a6')
    
    def run_controller(self):
        """运行手势控制器"""
        try:
            self.controller.run_camera()
        except Exception as e:
            messagebox.showerror("错误", f"运行过程中发生错误: {str(e)}")
        finally:
            self.running = False
            self.start_button.config(text="开始控制", bg='#27ae60')
            self.status_label.config(text="状态: 已停止")
            # 确保在退出时重置控制器状态
            self.controller.is_controlling = False
    
    def exit_app(self):
        """退出应用程序"""
        self.running = False
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()