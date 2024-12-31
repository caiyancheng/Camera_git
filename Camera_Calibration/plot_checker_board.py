import numpy as np
import cv2 as cv
from screeninfo import get_monitors


def get_second_screen_resolution():
    """
    获取第二个显示屏的分辨率。
    如果只有一个显示屏，则返回主显示屏的分辨率。
    :return: (width, height)
    """
    monitors = get_monitors()
    if len(monitors) > 1:
        # 返回第二个显示屏的分辨率
        monitor = monitors[1]
    else:
        # 如果只有一个显示屏，返回第一个显示屏的分辨率
        monitor = monitors[0]
    return monitor.width, monitor.height


def create_checkerboard(square_size=50, rows=6, cols=7, background_size=(800, 800), background_color=(128, 128, 128)):
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
    board_width = cols * square_size
    board_height = rows * square_size

    # 初始化背景图像
    background = np.ones((background_size[1], background_size[0], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # 创建棋盘格
    checkerboard = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 奇偶规则
                top_left = (j * square_size, i * square_size)
                bottom_right = ((j + 1) * square_size, (i + 1) * square_size)
                cv.rectangle(checkerboard, top_left, bottom_right, (0, 0, 0), -1)

    # 将棋盘格粘贴到背景的中心
    start_x = (background_size[0] - board_width) // 2
    start_y = (background_size[1] - board_height) // 2
    background[start_y:start_y + board_height, start_x:start_x + board_width] = checkerboard

    return background


# 参数设置
square_size = 50  # 每个格子的大小
rows, cols = 8, 7  # 棋盘格的行数和列数
# background_color = (255, 255, 255)  # 背景颜色，灰色
background_color = (128, 128, 128)  # 背景颜色，灰色

# 获取第二个显示屏的分辨率
background_size = get_second_screen_resolution()

# 创建带背景的棋盘格
checkerboard_image = create_checkerboard(square_size, rows, cols, background_size, background_color)

# 显示棋盘格
cv.imshow("Checkerboard", checkerboard_image)

# 保存到文件（可选）
cv.imwrite("checkerboard_fullscreen.jpg", checkerboard_image)

# 等待按键退出
cv.waitKey(0)
cv.destroyAllWindows()
