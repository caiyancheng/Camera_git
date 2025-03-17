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

# 获取第二个显示屏的分辨率和位置
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 设定中心区域的大小（可调）
display_width, display_height = 1000, 300

# 计算方块大小
num_blocks = 10  # 每行10个方块
block_width = display_width // num_blocks
block_height = display_height // 3  # 三行

# 创建黑色背景
background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# 计算显示区域左上角位置
start_x = (screen_width - display_width) // 2
start_y = (screen_height - display_height) // 2

# 生成颜色方块
for row in range(3):  # 三行（红、绿、蓝）
    for col in range(num_blocks):
        color = [0, 0, 0]  # 初始黑色
        color[row] = int((col / (num_blocks - 1)) * 255)  # 渐变颜色
        print(color)
        top_left = (start_x + col * block_width, start_y + row * block_height)
        bottom_right = (top_left[0] + block_width, top_left[1] + block_height)
        cv2.rectangle(background, top_left, bottom_right, color, -1)

# 创建窗口
window_name = "Color Blocks"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 显示图像
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
