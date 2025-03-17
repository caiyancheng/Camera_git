import cv2
import numpy as np
from screeninfo import get_monitors

# X-Rite ColorChecker Classic 24色的RGB值（需转换为BGR格式）
colorchecker_colors = [
    (68, 82, 115), (130, 150, 194), (157, 122, 98), (67, 108, 87),
    (177, 128, 133), (170, 189, 103), (44, 126, 214), (166, 91, 80),
    (99, 90, 193), (108, 60, 94), (64, 188, 157), (46, 163, 224),
    (150, 61, 56), (73, 148, 70), (60, 54, 175), (31, 199, 231),
    (149, 86, 187), (158, 166, 50), (242, 243, 242), (200, 200, 200),
    (160, 160, 160), (121, 122, 122), (85, 85, 85), (52, 52, 52)
]


# 获取第二个显示屏的分辨率和位置
def get_second_screen_resolution_and_position():
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]  # 使用第二个屏幕
    else:
        monitor = monitors[0]  # 只有一个屏幕时使用主屏幕
    return monitor.width, monitor.height, monitor.x, monitor.y


# 获取屏幕信息
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 矩形参数
rect_width = 2000 #00  # 矩形宽度
rect_height = 2000 #00  # 矩形高度

# 计算矩形位置，使其居中
rect_x = (screen_width - rect_width) // 2
rect_y = (screen_height - rect_height) // 2

# 创建窗口
window_name = "ColorChecker Display"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 初始化颜色索引
color_index = 0

while True:
    # 创建黑色背景
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    print(f'Showing Color {color_index}')

    # 获取当前颜色并填充矩形
    rect_color = colorchecker_colors[color_index]
    cv2.rectangle(background, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, -1)

    # 显示图像
    cv2.imshow(window_name, background)

    # 等待按键
    key = cv2.waitKey(0)

    if key == 32:  # 空格键（跳到下一个颜色）
        color_index = (color_index + 1) % len(colorchecker_colors)
    elif key == 27:  # ESC 键（退出）
        break

# 关闭窗口
cv2.destroyAllWindows()
