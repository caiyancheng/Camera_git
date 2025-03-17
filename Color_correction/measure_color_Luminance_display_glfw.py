import glfw
import numpy as np
import os
import time
import json
import threading
import pyautogui  # 用于模拟键盘输入
from OpenGL.GL import *
from screeninfo import get_monitors
from gfxdisp.specbos import specbos_measure, specbos_get_sprad
from tqdm import tqdm


def get_second_screen_resolution_and_position():
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]
    else:
        monitor = monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y


def measure_luminance(pixel_value, repeat_times, result_dict, event):
    time.sleep(1)
    Y_list, x_list, y_list, lmb_list, L_list = [], [], [], [], []
    for _ in range(repeat_times):
        Y, x, y = specbos_measure()
        lmb, L = specbos_get_sprad()
        Y_list.append(Y)
        x_list.append(x)
        y_list.append(y)
        lmb_list.append(lmb)
        L_list.append(L)
    result_dict["Y_mean"] = np.mean(Y_list, axis=0)
    result_dict["x_mean"] = np.mean(x_list, axis=0)
    result_dict["y_mean"] = np.mean(y_list, axis=0)
    event.set()
    pyautogui.press("space")


def render_rectangle(rgb_value, screen_width, screen_height):
    glClearColor(rgb_value[0] / 255.0, rgb_value[1] / 255.0, rgb_value[2] / 255.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)


def main():
    pixel_value_list = np.linspace(0, 255, 2).astype(np.uint8)
    json_dict = {}
    all_results = []
    Luminance_list = []
    screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()
    repeat_times = 2

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(screen_width, screen_height, "Color Display", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.set_window_pos(window, screen_x, screen_y)
    glfw.make_context_current(window)
    glfw.show_window(window)

    for pixel_value in tqdm(pixel_value_list):
        rgb_value = (int(pixel_value), int(pixel_value), int(pixel_value))
        render_rectangle(rgb_value, screen_width, screen_height)
        glfw.swap_buffers(window)

        result_dict = {}
        event = threading.Event()
        measure_thread = threading.Thread(target=measure_luminance,
                                          args=(pixel_value, repeat_times, result_dict, event))
        measure_thread.start()

        while not event.is_set():
            glfw.poll_events()

        print('pixel_value', pixel_value, 'Y_mean', result_dict["Y_mean"])
        Luminance_list.append(result_dict["Y_mean"])
        all_results.append({"pixel_value": int(pixel_value), **result_dict})

    glfw.destroy_window(window)
    glfw.terminate()

    json_dict['all_results'] = all_results
    json_dict['pixel_value_list'] = pixel_value_list.tolist()
    json_dict['Luminance_list'] = Luminance_list

    with open("color_luminance_measurements_rect_glfw.json", "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    main()
