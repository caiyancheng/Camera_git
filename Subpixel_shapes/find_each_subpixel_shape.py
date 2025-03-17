import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_subpixels(image_path, color):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 提取颜色通道
    if color == 'red':
        channel = image[:, :, 0]  # 红色通道
    elif color == 'green':
        channel = image[:, :, 1]  # 绿色通道
    elif color == 'blue':
        channel = image[:, :, 2]  # 蓝色通道
    elif color == 'white':
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度（白色）
    else:
        raise ValueError("Color must be 'red', 'green', 'blue', or 'white'")

    # 进行阈值分割
    _, binary = cv2.threshold(channel, 50, 255, cv2.THRESH_BINARY)

    # 发现轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    output = image.copy()
    cv2.drawContours(output, contours, -1, (255, 255, 0), 1)  # 黄色轮廓

    # 显示结果
    plt.figure(figsize=(8, 8))
    plt.imshow(output)
    plt.title(f'{color.capitalize()} Subpixels')
    plt.axis('off')
    plt.show()


# 处理四张图片
detect_subpixels('/mnt/data/DSC00000_PSMS.png', 'blue')
detect_subpixels('/mnt/data/DSC00004_PSMS.png', 'green')
detect_subpixels('/mnt/data/DSC00008_PSMS.png', 'red')
detect_subpixels('/mnt/data/DSC00016_PSMS.png', 'white')
