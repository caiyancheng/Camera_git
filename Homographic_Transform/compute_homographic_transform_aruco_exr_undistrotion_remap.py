import cv2
import numpy as np
import os
import json
import glob
import pyexr

camera_name = 'a7R3'
with open(f'../Camera_Calibration/calibration_result_SONY_{camera_name}_100cm_6_refine.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

# 设置Aruco字典和参数
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# 图案的行列、边长和间隔
rows = 14
cols = 14
marker_length = 24  # 标记的边长24个子像素间距（像素宽度）
marker_spacing = 24 # 标记之间的间隔24个子像素间距（像素宽度）
origin_x = (cols * marker_length + (cols - 1) * marker_spacing) / 2
origin_y = (rows * marker_length + (rows - 1) * marker_spacing) / 2
x_bias = 0
y_bias = 0
# Read Images
root_img_path = r'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process'
img_name_list = ['scale_2_aruco_MTF_vignetting_undistortion_LANCZOS4_updown_2.exr'] #100cm距离
img_path_list = [os.path.join(root_img_path, i) for i in img_name_list]

for img_path in img_path_list:
    obj_points = []  # 3D世界坐标点
    img_points = []  # 2D图像坐标点
    img_ID = img_path.split('\\')[-1].split('.')[0]
    exr = pyexr.open(img_path)
    exr_data = exr.get()
    exr_data[exr_data < 0] = 0
    image = (exr_data / 400.0 * 255).clip(0, 255).astype(np.uint8)
    # Undistortion Steps:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_scale = 1
    if resize_scale == 1:
        gray_resize = gray.copy()
    else:
        new_height = round(image.shape[0] / resize_scale)
        new_width = round(image.shape[1] / resize_scale)
        gray_resize = cv2.resize(gray, (new_width, new_height))
    blur_kernel_size = 11 #13是一个很好的数值
    gray_blur = cv2.GaussianBlur(gray_resize, (blur_kernel_size, blur_kernel_size), 0)
    cv2.imwrite(os.path.join(root_img_path, 'blur_undistort.png'), gray_blur)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray_blur)

    if ids is not None:
        for i, corner in enumerate(corners):
            corner *= resize_scale
            points = corner[0].astype(int)  # 将 corner 转换为整数像素坐标
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

            marker_id = ids[i][0]

            # 计算标记的行列位置
            row = marker_id // cols  # 计算行
            col = marker_id % cols  # 计算列

            # 计算该标记的中心点坐标
            center_x = col * (marker_length + marker_spacing) + marker_length / 2 - origin_x + x_bias
            center_y = (rows - row - 1) * (marker_length + marker_spacing) + marker_length / 2 - origin_y + y_bias

            # 计算标记四个角的坐标
            bias = 0 #1/2
            top_left = np.array([center_x - marker_length / 2 + bias, center_y + marker_length / 2 - bias, 0], dtype=np.float32)
            top_right = np.array([center_x + marker_length / 2 - bias, center_y + marker_length / 2 - bias, 0], dtype=np.float32)
            bottom_right = np.array([center_x + marker_length / 2 - bias, center_y - marker_length / 2 + bias, 0], dtype=np.float32)
            bottom_left = np.array([center_x - marker_length / 2 + bias, center_y - marker_length / 2 + bias, 0], dtype=np.float32)

            # 打印四个角的世界坐标
            # print(f"Marker ID: {marker_id}")
            # print(f"Top Left: {top_left}")
            # print(f"Top Right: {top_right}")
            # print(f"Bottom Right: {bottom_right}")
            # print(f"Bottom Left: {bottom_left}\n")

            # 计算姿态
            obj_points.extend([top_left, top_right, bottom_right, bottom_left])
            img_points.extend(corner[0])

            # 在标记中心绘制 ID
            center = tuple(corner[0].mean(axis=0).astype(int))
            cv2.putText(image, f"{marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 10, cv2.LINE_AA)
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length=100, thickness=3)
        json_data = {'obj_points': obj_points.tolist(), 'img_points': img_points.tolist(), 'retval': retval,
                     'rvec': rvec.tolist(), 'tvec': tvec.tolist()}
        with open(os.path.join(root_img_path, f'homography_aruco_result_{img_ID}_exr.json'), 'w') as fp:
            json.dump(json_data, fp)
    else:
        print("No markers detected.")

    # 显示结果
    output_path = f'Display_ArUco_together_{img_ID}_resize_{resize_scale}_blur_{blur_kernel_size}_exr_remap_INTER_LANCZOS4_updown_2.png'
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(root_img_path, output_path), img_bgr)
    print(os.path.join(root_img_path, output_path))
