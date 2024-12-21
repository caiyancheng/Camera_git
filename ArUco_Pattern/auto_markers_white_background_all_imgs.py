import tkinter as tk
from screeninfo import get_monitors
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

def get_second_screen_resolution():
    monitors = get_monitors()
    if len(monitors) < 2:
        raise Exception("未检测到第二个屏幕！请确保连接了多个显示器。")

    second_screen = monitors[1]
    return second_screen.width, second_screen.height, second_screen.x, second_screen.y

def generate_aruco_marker(pattern_id=0, marker_size=200, image_size=400):
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

    # 标记更新函数
    def update_marker():
        for pattern_id in range(250):
            marker_image = generate_aruco_marker(pattern_id=pattern_id)
            marker_pil = Image.fromarray(marker_image)
            marker_tk = ImageTk.PhotoImage(marker_pil)
            canvas.delete("all")
            canvas.create_image(width // 2, height // 2, image=marker_tk, anchor=tk.CENTER)
            canvas.image = marker_tk  # 防止图片被垃圾回收
            root.update()
            root.after(20*1000)  # 每两秒更新一次

    # 使用线程避免阻塞主循环
    threading.Thread(target=update_marker, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    create_marker_on_second_screen()
