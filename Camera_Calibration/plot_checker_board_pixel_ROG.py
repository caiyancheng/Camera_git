# 世界坐标使用pixel而非m为单位
import numpy as np
import cv2 as cv
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


def create_checkerboard(square_size_w=50, square_size_h=50, rows=6, cols=7, background_size=(800, 800), background_color=(128, 128, 128)):
    """
    创建一个居中显示的棋盘格图案，背景为指定颜色。

    :param square_size_w: 每个格子水平方向的像素大小。
    :param square_size_h: 每个格子垂直方向的像素大小。
    :param rows: 棋盘格的行数。
    :param cols: 棋盘格的列数。
    :param background_size: 背景图像的尺寸（宽度, 高度）。
    :param background_color: 背景颜色，默认为灰色。
    :return: 带背景的棋盘格图像。
    """
    # 棋盘格的整体尺寸
    board_width = cols * square_size_w
    board_height = rows * square_size_h

    # 初始化背景图像
    background = np.ones((background_size[1], background_size[0], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # 创建棋盘格（初始全部为白色）
    checkerboard = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 按奇偶规则填充黑色
                top_left = (j * square_size_w, i * square_size_h)
                # 注意：减 1 保证边长正好为 square_size_w 和 square_size_h 个像素
                bottom_right = ((j + 1) * square_size_w - 1, (i + 1) * square_size_h - 1)
                cv.rectangle(checkerboard, top_left, bottom_right, (0, 0, 0), -1)

    # 将棋盘格粘贴到背景的中心
    start_x = (background_size[0] - board_width) // 2
    start_y = (background_size[1] - board_height) // 2
    background[start_y:start_y + board_height, start_x:start_x + board_width] = checkerboard

    return background


# 参数设置
square_size_w = 24
square_size_h = 24
rows, cols = 42, 65  # 棋盘格的行数和列数
background_color = (64, 64, 64)  # 背景颜色，灰色

# 获取第二个显示屏的分辨率
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()
background_size = (screen_width, screen_height)

# 创建带背景的棋盘格
checkerboard_image = create_checkerboard(square_size_w, square_size_h, rows, cols, background_size, background_color)

window_name = "Checker Board"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# 显示棋盘格
cv.imshow(window_name, checkerboard_image)

# 保存到文件（可选）
# cv.imwrite("checkerboard_fullscreen.jpg", checkerboard_image)

# 等待按键退出
cv.waitKey(0)
cv.destroyAllWindows()
