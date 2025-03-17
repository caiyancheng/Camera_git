#世界坐标使用pixel而非m为单位
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

    :param square_size: 每个格子的像素大小（默认50x50像素）。
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

    # 创建棋盘格
    checkerboard = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 奇偶规则
                top_left = (j * square_size_w, i * square_size_h)
                bottom_right = ((j + 1) * square_size_w, (i + 1) * square_size_h)
                cv.rectangle(checkerboard, top_left, bottom_right, (0, 0, 0), -1)

    # 将棋盘格粘贴到背景的中心
    start_x = (background_size[0] - board_width) // 2
    start_y = (background_size[1] - board_height) // 2
    background[start_y:start_y + board_height, start_x:start_x + board_width] = checkerboard

    return background


# 参数设置
# square_size_w_m = 0.01  # 每个格子的大小 (m)
# square_size_h_m = 0.01  # 每个格子的大小 (m)
# display_width = 1.211
# display_width_pixel = 3840
# display_height = 0.681
# display_height_pixel = 2160
# square_size_w = round(square_size_w_m * display_width_pixel / display_width)
# square_size_h = round(square_size_h_m * display_height_pixel / display_height)
square_size_w = 32
square_size_h = 32
rows, cols = 20, 20  # 棋盘格的行数和列数
# background_color = (255, 255, 255)  # 背景颜色，灰色
background_color = (128, 128, 128)  # 背景颜色，灰色

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
