import os
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm

k = 0.001 # 最小单位为多少m？
# width_gain = 1.211
# height_gain = 0.681

camera_name = 'a7R1'
with open(f'../Camera_Calibration/calibration_result_SONY_{camera_name}_80cm.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

root_path = r'E:\sony_pictures\ArUco_Homo_1'
json_file_list = glob.glob(os.path.join(root_path,'homography_aruco_result_DSC*_undistorted.json'))

aligned_image_compute_list = []
for json_file in tqdm(json_file_list):
    json_file_ID = json_file.split('\\')[-1].split('.')[0].split('_')[-2]
    img_path = os.path.join(root_path, f'{json_file_ID}.png')
    with open(json_file, 'r') as fp:
        homography_aruco_result = json.load(fp)
    obj_points = np.array(homography_aruco_result['obj_points'])
    obj_points[:, 0] += 1.211 / 2
    obj_points[:, 1] = -obj_points[:, 1]
    obj_points[:, 1] += 0.681 / 2

    obj_points = obj_points / k
    img_points = np.array(homography_aruco_result['img_points'])
    display_points = obj_points[:, :2]
    homography_matrix, mask = cv2.findHomography(img_points, display_points)

    display_width = round(1.211 / k)
    display_height = round(0.681 / k)
    display_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    image = dst
    h, w = image.shape[:2]
    white_mask = np.ones([h, w, 3])

    aligned_white_mask = cv2.warpPerspective(white_mask, homography_matrix, (display_width, display_height))
    aligned_white_mask[aligned_white_mask == 0] = np.nan
    aligned_image = cv2.warpPerspective(image, homography_matrix, (display_width, display_height))
    aligned_image_compute = aligned_image * aligned_white_mask
    aligned_image_compute_list.append(aligned_image_compute)

    # cordinate_center_x = round(1.211 / 2 / k)
    # cordinate_center_y = round(0.681 / 2 / k)
    # radius = 20
    # cv2.circle(aligned_image, (cordinate_center_x, cordinate_center_y), radius, (0, 0, 255), -1)
    #
    # cv2.imwrite(os.path.join(root_path, f'aligned_{json_file_ID}.png'), aligned_image)
cordinate_center_x = round(1.211 / 2 / k)
cordinate_center_y = round(0.681 / 2 / k)
aligned_image_stack_array = np.stack(aligned_image_compute_list, axis=0)
aligned_image_average_array = np.nanmean(aligned_image_stack_array, axis=0)
cv2.imwrite(os.path.join(root_path, f'aligned_combine.png'), aligned_image_average_array)


