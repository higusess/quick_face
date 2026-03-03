# 文件名: quick_face_recognition.py
import cv2
import torch
import numpy as np
import os
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# 表情识别 - 使用 DeepFace 库（支持多种模型）
try:
    from deepface import DeepFace
    EMOPTION_ENABLED = True
except ImportError:
    print("警告: 未安装 deepface 库，表情识别功能不可用")
    print("安装命令: pip install deepface")
    EMOPTION_ENABLED = False

# 配置
FACE_DB_FILE = "face_database.json"  # 人脸数据库文件
FACE_IMG_DIR = "face_images"         # 保存人脸图片的目录

# 表情识别配置
EMOTION_DETECTOR = 'emotion'         # 使用 emotion 模型进行表情识别
EMOTION_BACKEND = 'opencv'           # 使用 opencv 后端（速度快）
# 可选后端: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
# 推荐使用: 'opencv'（最快）、'retinaface'（最准）、'mtcnn'（平衡）

# 表情中文映射
EMOTION_LABELS = {
    'angry': '😠 愤怒',
    'disgust': '🤢 厌恶',
    'fear': '😨 恐惧',
    'happy': '😊 开心',
    'sad': '😢 悲伤',
    'surprise': '😲 惊讶',
    'neutral': '😐 平静'
}

# 表情颜色映射
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # 红色
    'disgust': (0, 128, 0),    # 深绿
    'fear': (128, 0, 128),     # 紫色
    'happy': (0, 255, 255),    # 黄色
    'sad': (255, 0, 0),        # 蓝色
    'surprise': (0, 165, 255),  # 橙色
    'neutral': (128, 128, 128)  # 灰色
}

# 确保目录存在
os.makedirs(FACE_IMG_DIR, exist_ok=True)

# --- 1. 加载模型 (第一次运行会自动下载预训练权重) ---
print("正在加载模型...")
detector = YOLO('yolov8n.pt')  # 使用通用YOLOv8n模型
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
recognizer = InceptionResnetV1(pretrained='vggface2').eval()
print("模型加载完成！")

if EMOPTION_ENABLED:
    print(f"表情识别功能已启用 (使用 {EMOTION_BACKEND} 后端)")

# --- 2. 人脸数据库 ---
known_faces = {}


def load_face_database():
    """从文件加载人脸数据库"""
    global known_faces
    if os.path.exists(FACE_DB_FILE):
        try:
            with open(FACE_DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                known_faces = {k: np.array(v) for k, v in data.items()}
            print(f"已加载 {len(known_faces)} 个已注册人脸")
            return True
        except Exception as e:
            print(f"加载数据库失败: {e}")
    return False


def save_face_database():
    """保存人脸数据库到文件"""
    try:
        data = {k: v.tolist() for k, v in known_faces.items()}
        with open(FACE_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(known_faces)} 个人脸到数据库")
        return True
    except Exception as e:
        print(f"保存数据库失败: {e}")
        return False


def register_face_from_image(name, image_path):
    """从图片路径注册人脸"""
    global known_faces

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return False

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图片 - {image_path}")
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)

        if face is not None:
            embedding = recognizer(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            known_faces[name] = embedding
            save_face_database()
            print(f"已注册: {name}")
            return True
        else:
            print(f"注册失败: {image_path} 中未检测到人脸")
            return False
    except Exception as e:
        print(f"注册过程中出错: {e}")
        return False


def register_face_from_frame(name, frame):
    """从当前帧注册人脸"""
    global known_faces

    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)

        if face is not None:
            embedding = recognizer(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            known_faces[name] = embedding
            save_face_database()
            print(f"已注册: {name}")
            return True
        else:
            print(f"注册失败: 当前画面中未检测到人脸")
            return False
    except Exception as e:
        print(f"注册过程中出错: {e}")
        return False


def detect_emotion(face_img):
    """
    使用 DeepFace 检测人脸表情

    参数:
        face_img: 人脸图像 (BGR 格式)

    返回:
        (emotion, confidence) - 表情标签和置信度
    """
    if not EMOPTION_ENABLED:
        return ('neutral', 0.0)

    try:
        # 使用 DeepFace 进行表情识别
        result = DeepFace.analyze(
            face_img,
            actions=[EMOTION_DETECTOR],
            enforce_detection=False,  # 允许没有检测到人脸的情况
            detector_backend=EMOTION_BACKEND
        )

        # DeepFace 返回的是列表，取第一个结果
        if isinstance(result, list):
            result = result[0]

        # 获取表情信息
        emotions = result.get('emotion', {})
        if not emotions:
            return ('neutral', 0.0)

        # 找到置信度最高的表情
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        return (dominant_emotion, confidence)

    except Exception as e:
        # 表情识别失败，返回平静表情
        # print(f"表情识别错误: {e}")
        return ('neutral', 0.0)


# 初始化：加载已有的人脸数据库
load_face_database()

print("\n" + "="*50)
print("操作说明:")
print("  'q' - 退出程序")
print("  'r' - 注册当前画面的人脸 (输入姓名)")
print("  's' - 保存当前画面截图 (用于后续注册)")
print("  'e' - 开关表情识别功能")
print("="*50 + "\n")

# 全局变量
register_mode = False
input_name = ""
emotion_enabled = EMOPTION_ENABLED

# 表情缓存（避免频繁分析）
emotion_cache = {}
emotion_cache_ttl = 10  # 缓存10帧
frame_count = 0


# --- 3. 开始实时识别 ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误: 无法打开摄像头！")
    exit(1)

print("摄像头已打开，开始识别...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    frame_count += 1

    # Step A: YOLO人脸检测
    results = detector(frame, verbose=False)

    # 如果检测到结果
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            # 扩大检测框以包含完整的人脸
            h = y2 - y1
            w = x2 - x1
            x1 = max(0, x1 - int(w * 0.1))
            y1 = max(0, y1 - int(h * 0.3))
            x2 = min(frame.shape[1], x2 + int(w * 0.1))
            y2 = min(frame.shape[0], y2 + int(h * 0.1))

            # Step B: 裁剪人脸区域
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Step C: 用MTCNN对齐人脸
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_aligned = mtcnn(face_rgb)

            # Step D: 人脸识别
            name = "Unknown"
            color = (0, 0, 255)  # 红色表示未知

            if face_aligned is not None:
                embedding = recognizer(face_aligned.unsqueeze(0)).detach().cpu().numpy().flatten()

                if len(known_faces) > 0:
                    min_dist = float('inf')
                    for known_name, known_emb in known_faces.items():
                        dist = np.linalg.norm(embedding - known_emb)
                        if dist < min_dist:
                            min_dist = dist
                            name = known_name

                    if min_dist < 1.2:
                        color = (0, 255, 0)  # 绿色表示识别成功
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

            # Step E: 表情识别
            emotion = "neutral"
            emotion_conf = 0.0

            if emotion_enabled:
                # 使用位置作为缓存键
                cache_key = (x1, y1, x2, y2)

                # 检查缓存
                if cache_key in emotion_cache:
                    emotion, emotion_conf, last_frame = emotion_cache[cache_key]
                    if frame_count - last_frame < emotion_cache_ttl:
                        pass  # 使用缓存
                    else:
                        # 缓存过期，重新分析
                        emotion, emotion_conf = detect_emotion(face_crop)
                        emotion_cache[cache_key] = (emotion, emotion_conf, frame_count)
                else:
                    # 新人脸，分析表情
                    emotion, emotion_conf = detect_emotion(face_crop)
                    emotion_cache[cache_key] = (emotion, emotion_conf, frame_count)

            # 获取表情显示文本和颜色
            emotion_text = EMOTION_LABELS.get(emotion, emotion)
            emotion_color = EMOTION_COLORS.get(emotion, (128, 128, 128))

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制姓名
            cv2.putText(frame, name, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 绘制表情
            if emotion_enabled:
                # 表情文本
                emotion_label = f"{emotion_text} ({int(emotion_conf)}%)"
                cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

    # 显示状态信息
    status_y = 30
    status_texts = [
        f"表情识别: {'开启' if emotion_enabled else '关闭'}",
    ]

    for text in status_texts:
        cv2.putText(frame, text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status_y += 25

    # 显示注册模式提示
    if register_mode:
        cv2.rectangle(frame, (10, frame.shape[0] - 80), (400, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, f"Register: {input_name}", (20, frame.shape[0] - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "按 ENTER 确认, ESC 取消", (20, frame.shape[0] - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + FaceNet + 表情识别", frame)

    # 键盘事件处理
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("退出程序...")
        break

    elif key == ord('r'):
        register_mode = True
        input_name = ""
        print("\n请输入要注册的人名，然后按 ENTER 确认，按 ESC 取消")

    elif key == ord('e'):
        emotion_enabled = not emotion_enabled
        print(f"\n表情识别: {'开启' if emotion_enabled else '关闭'}")

    elif key == ord('s'):
        import time
        filename = f"{FACE_IMG_DIR}/snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"已保存截图: {filename}")

    elif register_mode:
        if key == 13:  # ENTER
            if input_name.strip():
                if register_face_from_frame(input_name.strip(), frame):
                    print(f"注册成功: {input_name.strip()}")
            register_mode = False
            input_name = ""
            print("退出注册模式")

        elif key == 27:  # ESC
            register_mode = False
            input_name = ""
            print("取消注册")

        elif key == 8 or key == 127:  # Backspace
            input_name = input_name[:-1]
        elif 32 <= key <= 126:  # 可打印字符
            input_name += chr(key)

cap.release()
cv2.destroyAllWindows()
print("程序结束")
