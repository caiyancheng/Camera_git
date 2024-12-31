import cv2
import numpy as np
import os

# 创建输出目录
output_dir = "Markers"
os.makedirs(output_dir, exist_ok=True)

# 设置Aruco标记的字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 设置标记参数
marker_size = 100  # 每个Aruco标记的边长（像素）
num_rows = 4       # 行数
num_cols = 8       # 列数
spacing = 50  # 标记之间的间距
margin = 50        # 边距大小

# 计算背景图像大小
background_width = num_cols * marker_size + (num_cols - 1) * spacing + 2 * margin
background_height = num_rows * marker_size + (num_rows - 1) * spacing + 2 * margin
background_color = 255  # 白色背景 (255代表白色)

# 创建白色背景
background = np.ones((background_height, background_width), dtype=np.uint8) * background_color

# 生成并放置Aruco标记
for row in range(num_rows):
    for col in range(num_cols):
        # 计算当前标记的索引
        marker_id = row * num_cols + col

        # 生成Aruco标记
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

        # 计算放置位置
        x_offset = margin + col * (marker_size + spacing)
        y_offset = margin + row * (marker_size + spacing)

        # 将Aruco标记放置到背景上
        background[y_offset:y_offset + marker_size, x_offset:x_offset + marker_size] = marker_image

# 保存最终图像
output_path = os.path.join(output_dir, "aruco_grid.png")
cv2.imwrite(output_path, background)

print(f"Aruco grid image saved to {output_path}")
