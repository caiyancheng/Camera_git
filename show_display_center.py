import tkinter as tk
import pygetwindow as gw
from screeninfo import get_monitors

def get_second_screen_resolution():
    monitors = get_monitors()
    if len(monitors) < 2:
        raise Exception("未检测到第二个屏幕！请确保连接了多个显示器。")

    second_screen = monitors[1]
    return second_screen.width, second_screen.height, second_screen.x, second_screen.y

def create_red_box_on_second_screen():
    width, height, left, top = get_second_screen_resolution()

    # 创建窗口
    root = tk.Tk()
    root.title("Red Box Center")

    # 设置窗口大小和位置
    root.geometry(f"{width}x{height}+{left}+{top}")

    # 全屏背景为白色
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack()

    # 红色方框大小
    box_size = 50
    center_x = width // 2
    center_y = height // 2

    # 绘制红色方框
    canvas.create_rectangle(
        center_x - box_size // 2, center_y - box_size // 2,
        center_x + box_size // 2, center_y + box_size // 2,
        fill="red"
    )

    root.mainloop()

if __name__ == "__main__":
    create_red_box_on_second_screen()
