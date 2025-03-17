import pyexr
import numpy as np
import cv2
import os
from tqdm import tqdm

# 读取 EXR 文件
def load_exr(filename):
    exr = pyexr.open(filename)
    data = exr.get()  # 读取所有通道
    return data


# 生成坐标网格并计算缩放坐标
def generate_scaled_coords(central_crop_width, central_crop_height, k_scale):
    x_coords = np.arange(central_crop_width)
    y_coords = np.arange(central_crop_height)
    scaled_x_coords = (x_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    scaled_y_coords = (y_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    return scaled_x_coords, scaled_y_coords


# 生成 mask（向量化处理）
def create_mask_from_exr(data, scaled_x, scaled_y, threshold, mask_shape):
    grid_x, grid_y = np.meshgrid(scaled_x, scaled_y, indexing='xy')
    values = np.mean(data[grid_y, grid_x], axis=-1)
    mask = (values > threshold).astype(np.uint8)
    return mask


# # 线性插值 mask 到 EXR 文件大小
# def resize_mask(mask, target_shape):
#     return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


# 主函数
if __name__ == "__main__":
    root_path = 'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process'
    exr_filename_list = ["CC_XYZ_Homo_Transform_color_fringing_letter_center_crop.exr",
                         "CC_XYZ_Homo_Transform_color_fringing_white_circle_center_crop.exr",
                         "CC_XYZ_Homo_Transform_color_fringing_black_circle_center_crop.exr",
                         "CC_XYZ_Homo_Transform_aruco_center_crop.exr"]
    threshold = 50
    k_scale = 5
    central_crop_width_pixel = int(3840 / 4)
    central_crop_height_pixel = int(2160 / 4)
    # 生成缩放坐标
    scaled_x_coords, scaled_y_coords = generate_scaled_coords(central_crop_width_pixel, central_crop_height_pixel,
                                                              k_scale)
    for exr_filename in tqdm(exr_filename_list):
        exr_filename_pure = exr_filename.split('.')[0]
        data = load_exr(os.path.join(root_path, exr_filename))
        mask = create_mask_from_exr(data, scaled_x_coords, scaled_y_coords, threshold,
                                    (central_crop_height_pixel, central_crop_width_pixel))
        cv2.imwrite(os.path.join(root_path, f"mask_{exr_filename_pure}.png"), mask * 255)  # 保存二值 mask
        # resized_mask = resize_mask(mask, data.shape[:2])
        # cv2.imwrite("resized_mask.png", resized_mask * 255)  # 保存插值后的 mask
