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


def create_checkerboard_with_circle(square_size_w=50, square_size_h=50, rows=6, cols=7, background_size=(800, 800),
                                    background_color=(128, 128, 128), disk_radius=1,
                                    circle_color=(255, 255, 255), checker_white_color=(255, 255, 255),
                                    checker_black_color=(0, 0, 0)):
    """
    创建一个居中显示的棋盘格图案，背景为指定颜色，并在中心添加一个圆形。

    :param square_size_w: 每个格子的像素大小（默认50像素）。
    :param square_size_h: 每个格子的像素大小（默认50像素）。
    :param rows: 棋盘格的行数。
    :param cols: 棋盘格的列数。
    :param background_size: 背景图像的尺寸（宽度, 高度）。
    :param background_color: 背景颜色，默认为灰色。
    :param disk_radius: 圆形的半径（单位：厘米）。
    :param circle_color: 圆形的填充颜色，默认为白色。
    :param checker_white_color: 棋盘格白色区域的颜色，默认为白色。
    :param checker_black_color: 棋盘格黑色区域的颜色，默认为黑色。
    :return: 带背景和圆形的棋盘格图像。
    """

    # 棋盘格的整体尺寸
    board_width = cols * square_size_w
    board_height = rows * square_size_h

    # 初始化背景图像
    background = np.ones((background_size[1], background_size[0], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # 创建棋盘格
    checkerboard = np.ones((board_height, board_width, 3), dtype=np.uint8) * checker_white_color
    checkerboard = checkerboard.astype(np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 奇偶规则
                top_left = (j * square_size_w, i * square_size_h)
                bottom_right = ((j + 1) * square_size_w, (i + 1) * square_size_h)
                cv.rectangle(checkerboard, top_left, bottom_right, checker_black_color, -1)

    # 将棋盘格粘贴到背景的中心
    start_x = (background_size[0] - board_width) // 2
    start_y = (background_size[1] - board_height) // 2
    background[start_y:start_y + board_height, start_x:start_x + board_width] = checkerboard

    # 计算圆形中心
    center_x = background_size[0] // 2
    center_y = background_size[1] // 2

    # 在图像中心添加圆形
    cv.circle(background, (center_x, center_y), disk_radius, circle_color, -1)

    return background


# 参数设置
square_size_w_m = 0.01  # 每个格子的大小 (m)
square_size_h_m = 0.01  # 每个格子的大小 (m)
disk_radius_m = 0.02
display_width = 1.211
display_width_pixel = 3840
display_height = 0.681
display_height_pixel = 2160
square_size_w = round(square_size_w_m * display_width_pixel / display_width)
square_size_h = round(square_size_h_m * display_height_pixel / display_height)
disk_radius = round(disk_radius_m * ((display_width_pixel / display_width) + (display_height_pixel / display_height))/2)
rows, cols = 20, 20  # 棋盘格的行数和列数
background_color = (128, 128, 128) # 背景颜色，灰色
disk_color = (128, 128, 128)
checker_white_color = (255, 255, 255)
checker_black_color = (0, 0, 0)

# 获取第二个显示屏的分辨率
background_size = get_second_screen_resolution()

# 创建带背景的棋盘格，并添加圆形
checkerboard_image = create_checkerboard_with_circle(square_size_w=square_size_w, square_size_h=square_size_h,
                                                     rows=rows, cols=cols, background_size=background_size,
                                                     background_color=background_color, disk_radius=disk_radius,
                                                     circle_color=disk_color,  # 圆形颜色为白色
                                                     checker_white_color=checker_white_color,  # 白色格子
                                                     checker_black_color=checker_black_color)  # 黑色格子

# 显示棋盘格
cv.imshow("Checkerboard with Circle", checkerboard_image)

# 保存到文件（可选）
# cv.imwrite("checkerboard_with_circle.jpg", checkerboard_image)

# 等待按键退出
cv.waitKey(0)
cv.destroyAllWindows()
