import cv2
import numpy as np
import os
import time
import json
import threading
import pyautogui  # 用于模拟键盘输入
from screeninfo import get_monitors
from gfxdisp.specbos import specbos_measure, specbos_get_sprad
from tqdm import tqdm


def get_second_screen_resolution_and_position():
    """
    获取第二个显示屏的分辨率和位置。
    :return: (width, height, x, y)
    """
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]
    else:
        monitor = monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y


def measure_luminance(pixel_value, repeat_times, result_dict, event):
    """
    在后台线程中进行测量，先等待 1 秒，确保屏幕图像已稳定，再进行测量。
    """
    time.sleep(1)  # 等待 1 秒，确保图像完全显示后再测量

    Y_list, x_list, y_list, lmb_list, L_list = [], [], [], [], []

    for _ in range(repeat_times):
        Y, x, y = specbos_measure()
        while Y == None:
            Y, x, y = specbos_measure()
        # lmb, L = specbos_get_sprad()
        Y_list.append(Y)
        x_list.append(x)
        y_list.append(y)
        # lmb_list.append(lmb)
        # L_list.append(L)

    result_dict["Y_mean"] = np.mean(Y_list, axis=0)
    result_dict["x_mean"] = np.mean(x_list, axis=0)
    result_dict["y_mean"] = np.mean(y_list, axis=0)
    # result_dict["lmb_mean"] = np.mean(lmb_list, axis=0).tolist()
    # result_dict["L_mean"] = np.mean(L_list, axis=0).tolist()

    event.set()  # 设定事件，通知主线程测量完成
    pyautogui.press("space")  # 模拟按键，立即结束 `cv2.waitKey(0)`


def main(rect_width, rect_height):
    pixel_num = 50
    pixel_value_list = np.linspace(0, 255, pixel_num).astype(np.uint8)
    # pixel_value_list = np.logspace(np.log10(1), np.log10(255), 50).astype(np.uint8)
    json_dict = {}
    all_results = []
    Luminance_list = []
    screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()
    if rect_width == rect_height == -1:
        rect_width = screen_width
        rect_height = screen_height
    repeat_times = 20

    for pixel_value in tqdm(pixel_value_list):
        rgb_value = (int(pixel_value), int(pixel_value), int(pixel_value))
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        rect_x, rect_y = (screen_width - rect_width) // 2, (screen_height - rect_height) // 2
        cv2.rectangle(background, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rgb_value, -1)

        window_name = "Color Display"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, screen_x, screen_y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, background)

        # 启动测量线程（测量线程会等待 1 秒）
        result_dict = {}
        event = threading.Event()
        measure_thread = threading.Thread(target=measure_luminance,
                                          args=(pixel_value, repeat_times, result_dict, event))
        measure_thread.start()

        # 显示图像并等待键盘输入，同时等待测量完成
        cv2.waitKey(0)  # `pyautogui.press("space")` 会自动触发这个函数结束
        event.wait()  # 确保测量线程完成

        print('pixel_value', pixel_value, 'Y_mean', result_dict["Y_mean"])
        Luminance_list.append(result_dict["Y_mean"])
        all_results.append({"pixel_value": int(pixel_value), **result_dict})

        cv2.destroyAllWindows()

    json_dict['all_results'] = all_results
    json_dict['pixel_value_list'] = pixel_value_list.tolist()
    json_dict['Luminance_list'] = Luminance_list
    save_root = f'Color_Luminance_Measure/pixel_num_{pixel_num}_repeat_{repeat_times}'
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, f"color_luminance_measurements_rect_width_{rect_width}_rect_height_{rect_height}.json"), "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    main(250, 250)
    main(500, 500)
    main(1000, 1000)
    main(2000, 2000)
    main(-1, -1)
