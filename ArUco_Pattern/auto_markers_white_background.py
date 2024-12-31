import tkinter as tk
from screeninfo import get_monitors
import cv2
import numpy as np
from PIL import Image, ImageTk

def get_second_screen_resolution():
    monitors = get_monitors()
    if len(monitors) < 2:
        raise Exception("未检测到第二个屏幕！请确保连接了多个显示器。")

    second_screen = monitors[1]
    return second_screen.width, second_screen.height, second_screen.x, second_screen.y

def generate_aruco_marker(pattern_id=100, marker_size=200, image_size=400):
    # 创建白色背景图像
    background_color = 255  # 白色背景 (255代表白色)

    # 创建一个全白的背景
    background = np.ones((image_size, image_size), dtype=np.uint8) * background_color

    # 设置Aruco标记的字典和ID
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # 生成Aruco标记图像
    marker_image = cv2.aruco.generateImageMarker(dictionary, pattern_id, marker_size)

    # 计算标记的放置位置（使其居中）
    x_offset = (background.shape[1] - marker_image.shape[1]) // 2
    y_offset = (background.shape[0] - marker_image.shape[0]) // 2

    # 将Aruco标记放置到白色背景上
    background[y_offset:y_offset + marker_image.shape[0], x_offset:x_offset + marker_image.shape[1]] = marker_image

    return background

def create_marker_on_second_screen():
    width, height, left, top = get_second_screen_resolution()

    # 创建窗口
    root = tk.Tk()
    root.title("Aruco Marker Center")

    # 设置窗口大小和位置
    root.geometry(f"{width}x{height}+{left}+{top}")

    # 全屏背景为白色
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack()

    # 生成Aruco标记
    marker_image = generate_aruco_marker()

    # 转换为PIL图像并居中显示
    marker_pil = Image.fromarray(marker_image)
    marker_tk = ImageTk.PhotoImage(marker_pil)
    canvas.create_image(width // 2, height // 2, image=marker_tk, anchor=tk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    create_marker_on_second_screen()
