# 人脸识别程序使用说明

## 功能特点

- 使用 YOLOv8n 进行人脸检测
- 使用 FaceNet (MTCNN + InceptionResnetV1) 进行人脸识别
- 支持实时人脸识别
- 支持运行时注册陌生人脸
- 人脸数据库可保存和迁移
- 终端运行，操作简单

## 安装依赖

```bash
pip install opencv-python torch numpy facenet-pytorch ultralytics
```

## 运行程序

```bash
python quick_face_recognition.py
```

## 操作说明

- `q` - 退出程序
- `r` - 注册当前画面的人脸（会提示输入姓名）
- `s` - 保存当前画面截图

## 人脸注册方法

### 方法1：运行时注册（推荐）
1. 打开程序
2. 让要注册的人面对摄像头
3. 按 `r` 键进入注册模式
4. 输入姓名
5. 按 `ENTER` 确认

### 方法2：从图片注册
在代码中添加：
```python
register_face_from_image("姓名", "path/to/image.jpg")
```

## 迁移到其他电脑

程序生成的文件（可以复制到其他电脑使用）：
- `face_database.json` - 人脸数据库（特征向量）
- `face_images/` - 保存的人脸图片（可选）

只需将这两个文件/文件夹与主程序一起复制到新电脑即可。

## 文件结构

```
quickface/
├── quick_face_recognition.py  # 主程序
├── face_database.json         # 人脸数据库（自动生成）
└── face_images/              # 人脸图片目录（自动生成）
```

## 注意事项

- 第一次运行会自动下载预训练模型（需要网络连接）
- 识别阈值可调整（默认1.2，可在代码中修改）
- 摄像头索引默认为0，如果有多个摄像头可修改代码中的 `cv2.VideoCapture(0)`
