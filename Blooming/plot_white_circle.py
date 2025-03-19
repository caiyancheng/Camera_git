import cv2
import numpy as np
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

# 设置显示屏的物理尺寸和分辨率
display_width_m = 1.211  # 显示屏宽度（米）
display_width_pixel = 3840  # 显示屏宽度（像素）
display_height_m = 0.681  # 显示屏高度（米）
display_height_pixel = 2160  # 显示屏高度（像素）

# 每米的像素数量
pixels_per_meter_w = display_width_pixel / display_width_m
pixels_per_meter_h = display_height_pixel / display_height_m
pixels_per_meter = (pixels_per_meter_w + pixels_per_meter_h) / 2

# 用户指定的圆形半径（以米为单位）
circle_radius_m = 0.05  # 圆形半径（米）

# 将半径转换为像素
circle_radius_pixel = int(circle_radius_m * pixels_per_meter)

# 获取第二个显示屏的分辨率和位置
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 创建黑色背景图像
background = np.zeros((screen_height, screen_width), dtype=np.uint8)

# 计算圆形的中心位置，使其居中
center_x = screen_width // 2
center_y = screen_height // 2

# 在背景上绘制白色圆形
cv2.circle(background, (center_x, center_y), circle_radius_pixel, 255, -1)  # -1 表示填充整个圆
background = background
# 创建窗口
window_name = "White Circle"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 显示图像
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
