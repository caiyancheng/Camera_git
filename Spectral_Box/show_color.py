import glfw
from OpenGL.GL import *
import sys

def create_fullscreen_window_on_second_screen(color):
    # 初始化 GLFW
    if not glfw.init():
        print("GLFW 初始化失败")
        sys.exit(1)

    # 获取监视器列表
    monitors = glfw.get_monitors()
    if len(monitors) < 2:
        print("只有一个显示屏，无法在第二屏幕显示！")
        glfw.terminate()
        return

    # 获取第二个屏幕的分辨率
    second_monitor = monitors[1]
    mode = glfw.get_video_mode(second_monitor)
    width, height = mode.size.width, mode.size.height

    # 创建全屏窗口在第二屏幕
    window = glfw.create_window(width, height, "Second Screen Color Display", second_monitor, None)
    if not window:
        print("窗口创建失败")
        glfw.terminate()
        sys.exit(1)

    # 设置 OpenGL 上下文
    glfw.make_context_current(window)

    # 主循环：填充颜色
    r, g, b = [c / 255.0 for c in color]  # 颜色值归一化到 0-1
    while not glfw.window_should_close(window):
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # 退出 GLFW
    glfw.destroy_window(window)
    glfw.terminate()

# 指定颜色 (1, 1, 1)
color = (255, 0, 0)  # RGB 值，接近黑色
create_fullscreen_window_on_second_screen(color)
