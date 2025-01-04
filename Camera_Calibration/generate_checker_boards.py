import numpy as np
import cv2

k = 1000
# 设置图像的尺寸和每个方块的边长 (单位: 像素)
# image_width = round((29.7 - 0.5 * 2) * k)
# image_height = round((21 - 0.5 * 2) * k)
image_width = 
square_size = 1 * k

# 计算棋盘的行列数
rows = 19 #image_height // square_size
cols = 28 #image_width // square_size

# 创建一个空的白色图像
checkerboard = np.ones((image_height, image_width), dtype=np.uint8) * 255

# 计算棋盘的起始位置（使棋盘居中）
start_x = (image_width - cols * square_size) // 2
start_y = (image_height - rows * square_size) // 2

# 填充棋盘格的黑色部分
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            checkerboard[start_y + i * square_size:start_y + (i + 1) * square_size,
                         start_x + j * square_size:start_x + (j + 1) * square_size] = 0

# 保存为图片
cv2.imwrite('checkerboard_centered.png', checkerboard)
