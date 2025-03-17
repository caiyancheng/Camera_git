import pyexr
import cv2
import numpy as np
import json
import random
from RGB_2_XYZ_color_correction import transform_RGB_2_XYZ_color_correction


# 读取EXR文件
def load_exr(exr_file_name):
    exr = pyexr.open(exr_file_name)
    return exr.get()


data = None
results = {}
results['All_input_RGB'] = []

def compute_mean(x, y, w, h):
    global data
    h, w = int(h), int(w)
    means = []
    for _ in range(5):  # 随机偏移 0~5 像素
        dx, dy = random.randint(-5, 5), random.randint(-5, 5)
        x1, y1 = max(0, x + dx), max(0, y + dy)
        x2, y2 = min(data.shape[1], x1 + w), min(data.shape[0], y1 + h)
        region = data[y1:y2, x1:x2]
        means.append(region.mean(axis=(0, 1)))
    return np.mean(means, axis=0)  # 计算所有矩形的均值


if __name__ == "__main__":
    exr_file = r"E:\sony_pictures\Color_Rects_Display_G1_whole_process_example/color_rects_0_4_8_12_merge_MTF_vignetting_ARQ.exr"  # 替换为你的EXR文件路径
    data = load_exr(exr_file)  # 读取 EXR 数据
    display_image = data[:, :, :3]  # 直接使用 float 数据，不归一化到 0-255
    scale_factor = 0.2  # 缩小显示比例
    display_resized = cv2.resize(display_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    display_resized = (display_resized / 200.0 * 255).clip(0, 255).astype(np.uint8)

    while True:
        roi = cv2.selectROI("Select Region", display_resized[:, :, ::-1], showCrosshair=True, fromCenter=False)
        if roi == (0, 0, 0, 0):
            break
        x, y, w, h = roi
        x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)

        mean_rgb_camera = compute_mean(x, y, w, h)
        mean_XYZ_linear = transform_RGB_2_XYZ_color_correction(mean_rgb_camera, expand=True)
        print(f"Selected Mean RGB Camera Linear: {mean_rgb_camera}")
        print(f"Selected Mean XYZ Linear: {mean_XYZ_linear}")

        # 在原始缩小图像上绘制选择的矩形并显示均值信息
        cv2.rectangle(display_resized, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        text_position = (roi[0], max(roi[1] - 10, 10))
        text_rgb = f"RGB: {mean_rgb_camera[0]:.1f}, {mean_rgb_camera[1]:.1f}, {mean_rgb_camera[2]:.1f}"
        text_xyz = f"XYZ: {mean_XYZ_linear[0]:.1f}, {mean_XYZ_linear[1]:.1f}, {mean_XYZ_linear[2]:.1f}"
        cv2.putText(display_resized, text_rgb, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(display_resized, text_xyz, (text_position[0], text_position[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 1, cv2.LINE_AA)

        user_rgb = input("Enter RGB values (comma-separated): ")
        try:
            r, g, b = map(int, user_rgb.split(','))
            results['All_input_RGB'].append([r,g,b])
            results[f"r{r}, g{g}, b{b}"] = {"Camera Mean RGB Linear": mean_rgb_camera.tolist(),
                                  "Camera Mean XYZ Linear": mean_XYZ_linear.tolist()}
        except ValueError:
            print("Invalid input, skipping...")

    cv2.destroyAllWindows()

    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to output.json")