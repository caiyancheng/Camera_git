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

# 设置字母组合的尺寸和间距
font_scale = 5  # 字母的大小
font_thickness = 10  # 字母的厚度
spacing_m = 0.05  # 字母之间的间距（米）

# 将间距转换为像素
spacing_pixel = int(spacing_m * pixels_per_meter)

# 获取第二个显示屏的分辨率和位置
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# 创建黑色背景图像
background = np.zeros((screen_height, screen_width), dtype=np.uint8)

# 设置字母的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 生成字母（A, B, C, D）的图像并计算每个字母的宽度和高度
letters = ['A', 'B', 'C', 'D']
row_num = 2
col_num = 2
if row_num * col_num != len(letters):
    raise ValueError('Rows Number and Columns Number may not be accurate!')

letter_images = []
max_width = 0
max_height = 0
for letter in letters:
    text_size = cv2.getTextSize(letter, font, font_scale, font_thickness)[0]
    letter_images.append(text_size)
    max_width = max(max_width, text_size[0])
    max_height = max(max_height, text_size[1])

# 计算字母中心的放置位置，使其在屏幕上居中
center_x = screen_width // 2
center_y = screen_height // 2

# 计算整个字母组合的总宽度和高度
total_width = max_width * col_num + spacing_pixel
total_height = max_height * row_num + spacing_pixel

# 计算字母组合的左上角位置，使其中心与屏幕中心对齐
start_x = center_x - total_width // 2
start_y = center_y - total_height // 2

# 在背景图像上绘制字母
current_y = start_y
for i, letter in enumerate(letters):
    row = i // 2  # 计算行号
    col = i % 2  # 计算列号

    # 计算每个字母的放置位置
    x_offset = start_x + col * (max_width + spacing_pixel)
    y_offset = current_y + row * (letter_images[i][1] + spacing_pixel)

    # 绘制字母
    cv2.putText(background, letter, (x_offset, y_offset + letter_images[i][1]), font, font_scale, (255), font_thickness)

# 创建窗口
window_name = "Letter Grid"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # 将窗口移动到第二屏
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 显示图像
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
