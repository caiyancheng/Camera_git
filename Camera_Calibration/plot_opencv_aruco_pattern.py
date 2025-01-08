import cv2
import numpy as np
import os
from screeninfo import get_monitors

def get_second_screen_resolution_and_position():
    """
    获取第二个显示屏的分辨率和位置。
    如果只有一个显示屏，则返回主显示屏的分辨率和位置。
    :return: (width, height, x, y)
    """
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]
    else:
        monitor = monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y

# 创建输出目录
output_dir = "Markers"
os.makedirs(output_dir, exist_ok=True)

# 设置Aruco标记的字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 显示屏参数
display_width_m = 1.211  # 显示屏宽度（米）
display_width_pixel = 3840  # 显示屏宽度（像素）
display_height_m = 0.681  # 显示屏宽度（米）
display_height_pixel = 2160  # 显示屏宽度（像素）
pixels_per_meter_w = display_width_pixel / display_width_m  # 每米的像素数量
pixels_per_meter_h = display_height_pixel / display_height_m
pixels_per_meter = (pixels_per_meter_w + pixels_per_meter_h) / 2

# 设置标记参数
marker_size_m = 0.03  # 每个Aruco标记的边长（米）
marker_size_w = round(marker_size_m * pixels_per_meter)  # 标记边长（像素）
spacing_m = 0.03  # 标记之间的间距（米）
spacing = round(spacing_m * pixels_per_meter)  # 间距（像素）

num_rows = int((display_height_pixel + spacing) / (marker_size_w + spacing))  # 行数
num_cols = int((display_width_pixel + spacing) / (marker_size_w + spacing))  # 列数
# margin_m = 0.01  # 边距大小（米）
# margin = round(margin_m * pixels_per_meter)  # 边距（像素）

# 获取第二个显示屏的分辨率和位置
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 创建与第二屏幕分辨率匹配的白色背景
background = np.ones((screen_height, screen_width), dtype=np.uint8) * 255

# 计算网格总宽高
grid_width = num_cols * marker_size_w + (num_cols - 1) * spacing
grid_height = num_rows * marker_size_w + (num_rows - 1) * spacing

# 确保网格居中
x_start = (screen_width - grid_width) // 2
y_start = (screen_height - grid_height) // 2

# 生成并放置Aruco标记
for row in range(num_rows):
    for col in range(num_cols):
        # 计算当前标记的索引
        marker_id = row * num_cols + col
        # if marker_id >= 250:
        #     marker_id = marker_id % 250

        # 生成Aruco标记
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_w)

        # 计算标记放置位置
        x_offset = x_start + col * (marker_size_w + spacing)
        y_offset = y_start + row * (marker_size_w + spacing)

        # 将Aruco标记放置到背景上
        background[y_offset:y_offset + marker_size_w, x_offset:x_offset + marker_size_w] = marker_image

# 保存最终图像
output_path = os.path.join(output_dir, "aruco_grid_screen.png")
cv2.imwrite(output_path, background)

# 创建窗口
window_name = "Aruco Grid"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 显示图像
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
