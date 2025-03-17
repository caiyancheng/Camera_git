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

# 获取第二个显示屏的分辨率和位置
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 设置矩形参数
rect_width = 200  # 矩形宽度（像素）
rect_height = 200  # 矩形高度（像素）
rect_color = (255, 255, 255)  # 矩形颜色

# 创建黑色背景
background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# 计算矩形位置，使其居中
rect_x = (screen_width - rect_width) // 2
rect_y = (screen_height - rect_height) // 2

# 在背景上绘制矩形
cv2.rectangle(background, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, -1)

# 创建窗口
window_name = "Central Rectangle"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 显示图像
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
