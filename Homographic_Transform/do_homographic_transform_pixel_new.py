import os
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm

# 放大多少倍 （3*3的图像代表之前的一个像素,每两个相邻像素共同占有一个像素）
k_scale = 5 #最小为3，且必须为整奇数
display_width_pixel = 3840
display_height_pixel = 2160
camera_name = 'a7R3'
with open(f'../Camera_Calibration/calibration_result_SONY_{camera_name}_60cm.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

root_path = r'E:\sony_pictures\a7R3_100_aruco_4_demosaic_60_distance'
json_file_list = glob.glob(os.path.join(root_path,'homography_aruco_result_DSC*_PSMS.json'))

aligned_image_compute_list = []
for json_file in tqdm(json_file_list):
    json_file_ID = json_file.split('\\')[-1].split('.')[0].split('_')[-2] + '_' + json_file.split('\\')[-1].split('.')[0].split('_')[-1]
    img_path = os.path.join(root_path, f'{json_file_ID}.png')
    with open(json_file, 'r') as fp:
        homography_aruco_result = json.load(fp)
    obj_points = np.array(homography_aruco_result['obj_points'])
    obj_points[:, 0] += display_width_pixel / 2 # - 1/2 #将起始位置由中点放到左上角，并反转y轴坐标,注意由于Homographic Transform时候是以像素间为中心的，所以这里要少1/2
    obj_points[:, 1] = -obj_points[:, 1]
    obj_points[:, 1] += display_height_pixel / 2 # - 1/2 #obj_pints位于两个像素之间（由homographic transform的定义可以知道）

    obj_points = obj_points * (k_scale-1)
    img_points = np.array(homography_aruco_result['img_points'])
    display_points = obj_points[:, :2]
    homography_matrix, mask = cv2.findHomography(img_points, display_points)

    display_image_record_width = round(display_width_pixel * (k_scale-1) + 1)
    display_image_record_height = round(display_height_pixel * (k_scale-1) + 1)
    display_image = np.zeros((display_image_record_height, display_image_record_width, 3), dtype=np.uint8)

    image = cv2.imread(img_path)
    # h, w = image.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    # image = dst
    h, w = image.shape[:2]
    white_mask = np.ones([h, w, 3])

    aligned_white_mask = cv2.warpPerspective(white_mask, homography_matrix, (display_image_record_width, display_image_record_height))
    aligned_white_mask[aligned_white_mask == 0] = np.nan
    aligned_image = cv2.warpPerspective(image, homography_matrix, (display_image_record_width, display_image_record_height))
    aligned_image_compute = aligned_image * aligned_white_mask
    aligned_image_compute_list.append(aligned_image_compute)
aligned_image_stack_array = np.stack(aligned_image_compute_list, axis=0)
aligned_image_average_array = np.nanmean(aligned_image_stack_array, axis=0)

# 计算所有符合条件的索引
x_coords = np.arange(display_width_pixel)  # 生成所有原始 x 坐标
y_coords = np.arange(display_height_pixel)  # 生成所有原始 y 坐标
scaled_x_coords = (x_coords * (k_scale-1) + (k_scale-1)/2).astype(int) # 计算缩放后的 x 坐标
scaled_y_coords = (y_coords * (k_scale-1) + (k_scale-1)/2).astype(int)  # 计算缩放后的 y 坐标
# 生成网格索引
scaled_x_grid, scaled_y_grid = np.meshgrid(scaled_x_coords, scaled_y_coords, indexing='ij')
# 确保索引在合法范围内
valid_x = scaled_x_grid < aligned_image_average_array.shape[1]
valid_y = scaled_y_grid < aligned_image_average_array.shape[0]
valid_indices = valid_x & valid_y  # 逻辑与，确保 x 和 y 坐标都在范围内
# 将所有满足条件的点变红
aligned_image_average_array[scaled_y_grid[valid_indices], scaled_x_grid[valid_indices]] = [0, 0, 255]
print('Start CV2 Writing')
cv2.imwrite(os.path.join(root_path, f'aligned_combine.png'), aligned_image_average_array)


