import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
import json

square_size = 16
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
checkerboard_size_row = 41
checkerboard_size_col = 64
x, y = np.mgrid[0:checkerboard_size_col, 0:checkerboard_size_row]
objp = np.zeros((checkerboard_size_row * checkerboard_size_col, 3), np.float32)
objp[:, :2] = np.column_stack((x.ravel(), y.ravel()))
objp[:, 0] *= square_size
objp[:, 1] *= square_size
center_x = (np.max(objp[:, 0]) + np.min(objp[:, 0])) / 2
center_y = (np.max(objp[:, 1]) + np.min(objp[:, 1])) / 2
objp[:, 0] -= center_x
objp[:, 1] -= center_y

objpoints = []
imgpoints = []


def fast_peak_local_max(image, min_distance=3, threshold_abs=50):
    # 应用阈值过滤
    mask_above_thresh = image >= threshold_abs
    if not np.any(mask_above_thresh):
        return np.empty((0, 2), dtype=np.intp)
    # 创建结构元素（矩形对应棋盘距离）
    kernel_size = 2 * min_distance + 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 膨胀操作
    dilated = cv2.dilate(image, kernel)
    # 候选点：原图等于膨胀结果且超过阈值
    coords = np.column_stack(np.where((image == dilated) & mask_above_thresh))
    # coords = np.column_stack(np.where((image == dilated) & mask_above_thresh))
    if len(coords) == 0:
        return coords
    # 按强度降序排序
    intensities = image[coords[:, 0], coords[:, 1]]
    sorted_indices = np.argsort(-intensities)
    coords = coords[sorted_indices]

    mask = np.zeros(image.shape, dtype=bool)
    selected = []

    # 滑动窗口半径
    y_min = np.maximum(0, coords[:, 0] - min_distance)
    y_max = np.minimum(image.shape[0], coords[:, 0] + min_distance + 1)
    x_min = np.maximum(0, coords[:, 1] - min_distance)
    x_max = np.minimum(image.shape[1], coords[:, 1] + min_distance + 1)

    # 向量化处理
    for i in range(len(coords)):
        y, x = coords[i]
        if not mask[y, x]:
            selected.append([y, x])
            # 向量化屏蔽操作
            mask[y_min[i]:y_max[i], x_min[i]:x_max[i]] = True

    return np.array(selected)

distance = '100cm_6'
image_root_path = rf'E:\sony_pictures\Calibration_a7R3_on_display\{distance}'
save_clibrate_image_path = image_root_path + '_calibrate_refine_2'
os.makedirs(save_clibrate_image_path, exist_ok=True)
images = glob.glob(os.path.join(image_root_path, 'DSC*.png'))
# images = [os.path.join(image_root_path, 'DSC00002.png')]
save_images = True
for fname in tqdm(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = 1
    blur_kernel_size_2 = 5
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    gray_blur_2 = cv2.GaussianBlur(gray, (blur_kernel_size_2, blur_kernel_size_2), 0)
    coordinates = fast_peak_local_max(gray_blur_2, min_distance=3, threshold_abs=100)
    if len(coordinates) != (checkerboard_size_row+1) * (checkerboard_size_col+1) * square_size * square_size / 2:
        raise ValueError('Have not found correct coordinates!')
    coordinates_mask = np.zeros_like(gray_blur_2, dtype=np.uint8)
    coordinates_mask[coordinates[:, 0], coordinates[:, 1]] = 1
    ret, corners = cv2.findChessboardCorners(gray_blur, (checkerboard_size_row, checkerboard_size_col), None)
    if ret == True:
        search_range = 20
        # largest_move = 4
        adjusted_corners = corners.copy()
        for corner_index in range(len(corners)):
            corner = corners[corner_index].ravel()
            col_num = corner_index // checkerboard_size_row
            order_num = corner_index % checkerboard_size_row
            mask = coordinates_mask[
                   max(0, int(corner[1]) - search_range):min(gray.shape[0], int(corner[1]) + search_range),
                   max(0, int(corner[0]) - search_range):min(gray.shape[1], int(corner[0]) + search_range)]
            y_idx, x_idx = np.where(mask == 1)
            if len(x_idx) > 1 and len(y_idx) > 1:
                center = np.array([mask.shape[1] // 2, mask.shape[0] // 2])
                candidate_points = np.column_stack((x_idx, y_idx))
                distances = np.linalg.norm(candidate_points - center, axis=1)
                # 选出最近的一个点
                nearest_index = np.argmin(distances)
                nearest_point = candidate_points[nearest_index]
                # 剔除所有与最近点在x或y方向距离小于2的点
                near_mask = np.logical_and(np.abs(candidate_points[:, 0] - nearest_point[0]) >= 2,
                                     np.abs(candidate_points[:, 1] - nearest_point[1]) >= 2)
                filtered_points = candidate_points[near_mask]
                filtered_distances = np.linalg.norm(filtered_points - center, axis=1)
                second_nearest_index = np.argmin(filtered_distances)
                second_nearest_point = filtered_points[second_nearest_index]
                nearest_two = np.array([nearest_point, second_nearest_point])

                offset = np.mean(nearest_two, axis=0) - center
                adjusted_corners[corner_index] += offset
                # diff = nearest_two - center
                # if (np.sign(diff[0, 0]) != np.sign(diff[1, 0])) and (np.sign(diff[0, 1]) != np.sign(diff[1, 1])):
                #     offset = np.mean(nearest_two, axis=0) - center
                #     adjusted_corners[corner_index] += offset
                # else:
                #     print('Wrong Points!')
        objpoints.append(objp)
        imgpoints.append(adjusted_corners)

        if save_images:
            cv2.drawChessboardCorners(img, (checkerboard_size_row, checkerboard_size_col), adjusted_corners, ret)
            cv2.imwrite(os.path.join(save_clibrate_image_path, fname.split('\\')[-1]), img)
    else:
        print('Cannot find the checker board!')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))
re_projection_error = mean_error / len(objpoints)
json_data = {'ret': ret, 'mtx': mtx.tolist(), 'dist': dist.tolist(), 're_projection_error': re_projection_error}
with open(f'calibration_result_SONY_a7R3_{distance}_refine_2.json', 'w') as outfile:
    json.dump(json_data, outfile)
