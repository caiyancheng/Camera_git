import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_grid_lines(image_path, threshold=100, min_line_length=10, max_line_gap=10):
    # 读取图像并转换为灰度
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 使用 Hough 变换检测直线
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # 仅保留水平和垂直方向的线条
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 计算角度
            if abs(angle) < 5 or abs(angle - 90) < 5:  # 只保留接近 0° 或 90° 的直线
                filtered_lines.append([x1, y1, x2, y2])

    return img, filtered_lines


def draw_lines(image, lines, color=(0, 255, 0), thickness=1):
    output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(output_img, (x1, y1), (x2, y2), color, thickness)
    return output_img


# 运行检测并绘制网格
image_path = r"E:\sony_pictures\Calibration_a7R3_on_display\50cm/DSC00001.png"
img, grid_lines = detect_grid_lines(image_path)
output_img = draw_lines(img, grid_lines)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(output_img)
plt.axis('off')
plt.show()

# 保存输出
output_path = r"E:\sony_pictures\Calibration_a7R3_on_display\50cm/DSC00001_hough_grid.png"
cv2.imwrite(output_path, output_img)