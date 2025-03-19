import os
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm
import re
import pyexr
from Color_correction.RGB_2_XYZ_color_correction import transform_RGB_2_XYZ_color_correction
from Color_space_Transform import cm_xyz2rgb

# 放大多少倍 （3*3的图像代表之前的一个像素,每两个相邻像素共同占有一个像素）
k_scale = 5 #最小为3，且必须为整奇数
display_width_pixel = 3840
display_height_pixel = 2160
central_crop_width_pixel = display_width_pixel / 4
central_crop_height_pixel = display_height_pixel / 4

camera_name = 'a7R3'
with open(f'../Camera_Calibration/calibration_result_SONY_{camera_name}_100cm_6_refine.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

root_path = r'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process_new'
save_path = r'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process_new'
if not os.path.exists(save_path):
    os.makedirs(save_path)
json_file_name = 'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process/homography_aruco_result_aruco_MTF_vignetting_undistortion_remap_INTER_LANCZOS4_exr.json'
# image_list = ['color_fringing_letter.exr']
image_list = ['color_fringing_letter_MTF_vignetting_undistortion_remap_INTER_LANCZOS4.exr',
              'color_fringing_white_circle_MTF_vignetting_undistortion_remap_INTER_LANCZOS4.exr',
              'color_fringing_black_circle_MTF_vignetting_undistortion_remap_INTER_LANCZOS4.exr']
with open(os.path.join(root_path, json_file_name), 'r') as fp:
    homography_aruco_result = json.load(fp)
obj_points = np.array(homography_aruco_result['obj_points'])
obj_points[:, 0] = obj_points[:, 0] + display_width_pixel / 2  # - 1/2 #将起始位置由中点放到左上角，并反转y轴坐标,注意由于Homographic Transform时候是以像素间为中心的，所以这里要少1/2
obj_points[:, 1] = -obj_points[:, 1]
obj_points[:, 1] = obj_points[:, 1] + display_height_pixel / 2  # - 1/2 #obj_pints位于两个像素之间（由homographic transform的定义可以知道）

obj_points = obj_points * (k_scale - 1)
img_points = np.array(homography_aruco_result['img_points'])
display_points = obj_points[:, :2]
homography_matrix, mask = cv2.findHomography(img_points, display_points)

for img_name in tqdm(image_list):
    img_path = os.path.join(root_path, img_name)
    img_pure_name = img_name.split('.')[0]
    display_image_record_width = round(display_width_pixel * (k_scale - 1) + 1)
    display_image_record_height = round(display_height_pixel * (k_scale - 1) + 1)
    central_crop_display_image_record_width = round(central_crop_width_pixel * (k_scale - 1) + 1)
    central_crop_display_image_record_height = round(central_crop_height_pixel * (k_scale - 1) + 1)
    display_image = np.zeros((display_image_record_height, display_image_record_width, 3), dtype=np.uint8)

    exr = pyexr.open(img_path)
    exr_data = exr.get()
    exr_data[exr_data < 0] = 0
    exr_data_XYZ_linear = transform_RGB_2_XYZ_color_correction(exr_data, expand=True)
    # image = (exr_data / 1000.0 * 255).clip(0, 255).astype(np.uint16)
    exr_data_XYZ_linear[exr_data_XYZ_linear < 0] = 0
    exr_data_XYZ_linear = exr_data_XYZ_linear.astype(np.float32)
    h, w = exr_data_XYZ_linear.shape[:2]
    aligned_image = cv2.warpPerspective(exr_data_XYZ_linear, homography_matrix,
                                        (display_image_record_width, display_image_record_height))
    ##对应区域打上红点
    x_coords = np.arange(display_width_pixel)
    y_coords = np.arange(display_height_pixel)
    scaled_x_coords = (x_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    scaled_y_coords = (y_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    scaled_x_grid, scaled_y_grid = np.meshgrid(scaled_x_coords, scaled_y_coords, indexing='ij')
    valid_x = scaled_x_grid < aligned_image.shape[1]
    valid_y = scaled_y_grid < aligned_image.shape[0]
    valid_indices = valid_x & valid_y
    # aligned_image[scaled_y_grid[valid_indices], scaled_x_grid[valid_indices]] = [400, 0, 0]

    start_x = (display_image_record_width - central_crop_display_image_record_width) // 2
    start_y = (display_image_record_height - central_crop_display_image_record_height) // 2
    cropped_image_xyz_linear = aligned_image[start_y:start_y + central_crop_display_image_record_height,
                               start_x:start_x + central_crop_display_image_record_width]
    cropped_image_rgb_linear = cm_xyz2rgb(cropped_image_xyz_linear, 'sRGB')
    cropped_image_rgb_linear[cropped_image_rgb_linear<0] = 0
    cropped_image_rgb_encoded = (cropped_image_rgb_linear / 400) ** (1 / 2.2)
    cropped_image_rgb_clipped = np.clip(cropped_image_rgb_encoded, 0, 1) * 255
    cropped_image_rgb_uint8 = cropped_image_rgb_clipped.round().astype(np.uint8)
    # img_bgr = cv2.cvtColor(aligned_image_average_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, f'CC_sRGB_Homo_Transform_{img_pure_name}_center_crop_undistortion_remap_{k_scale}.png'), cropped_image_rgb_uint8[:, :, ::-1])
    out_path = os.path.join(save_path, f'CC_XYZ_Homo_Transform_{img_pure_name}_center_crop_undistortion_remap_{k_scale}.exr')
    # np.savez_compressed(os.path.join(save_path, f"CC_XYZ_Homo_Transform_{img_pure_name}_center_crop.npz"), cropped_image=cropped_image_xyz_linear)
    pyexr.write(out_path, cropped_image_xyz_linear)

