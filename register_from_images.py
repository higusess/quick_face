# 批量注册人脸脚本
import cv2
import torch
import numpy as np
import os
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1

# 从主程序导入函数
sys.path.insert(0, os.path.dirname(__file__))
from quick_face_recognition import load_face_database, save_face_database, known_faces

# 加载模型
print("正在加载模型...")
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
recognizer = InceptionResnetV1(pretrained='vggface2').eval()
print("模型加载完成！")

# 加载已有数据库
load_face_database()


def register_from_folder(folder_path):
    """从文件夹批量注册人脸"""
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return

    # 支持的图片格式
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    files = os.listdir(folder_path)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print(f"在 {folder_path} 中未找到图片文件")
        return

    print(f"\n找到 {len(image_files)} 张图片")
    print("="*50)

    success_count = 0
    fail_count = 0

    for img_file in image_files:
        # 从文件名提取人名（去掉扩展名）
        name = os.path.splitext(img_file)[0]
        img_path = os.path.join(folder_path, img_file)

        print(f"\n处理: {img_file} -> 人名: {name}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  错误: 无法读取图片")
                fail_count += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)

            if face is not None:
                embedding = recognizer(face.unsqueeze(0)).detach().cpu().numpy().flatten()

                if name in known_faces:
                    print(f"  警告: {name} 已存在，将被覆盖")
                else:
                    print(f"  成功: 已注册 {name}")

                known_faces[name] = embedding
                success_count += 1
            else:
                print(f"  失败: 未检测到人脸")
                fail_count += 1

        except Exception as e:
            print(f"  错误: {e}")
            fail_count += 1

    # 保存数据库
    save_face_database()

    print("\n" + "="*50)
    print(f"批量注册完成！")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"当前总人数: {len(known_faces)}")


if __name__ == "__main__":
    print("="*50)
    print("批量人脸注册工具")
    print("="*50)
    print("\n用法: python register_from_images.py <图片文件夹>")
    print("\n说明:")
    print("  - 将图片放入指定文件夹")
    print("  - 图片文件名将作为人名（如: 张三.jpg -> 人名: 张三）")
    print("  - 支持格式: .jpg, .jpeg, .png, .bmp")

    if len(sys.argv) < 2:
        folder = input("\n请输入图片文件夹路径: ").strip()
    else:
        folder = sys.argv[1]

    register_from_folder(folder)
