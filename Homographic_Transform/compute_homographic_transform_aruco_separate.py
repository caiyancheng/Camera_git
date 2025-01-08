import cv2
import numpy as np
import os
import json

camera_name = 'a7R1'
with open(f'../Camera_Calibration/calibration_result_SONY_{camera_name}.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

# 设置Aruco字典和参数
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# 准备标定所需的变量
obj_points = []  # 3D世界坐标点
img_points = []  # 2D图像坐标点

# Read Images
img_path = r'E:\sony_pictures\Homographic_1/DSC00002.png'
image = cv2.imread(img_path)

# #Undistortion Process:
# h,  w = image.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
# dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# image = dst

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resize_scale = 2
new_height = round(image.shape[0] / resize_scale)
new_width = round(image.shape[1] / resize_scale)
gray_resize = cv2.resize(gray, (new_width, new_height))

corners, ids, rejectedImgPoints = detector.detectMarkers(gray_resize)
marker_length = 0.02 #不好意思没有衡量清楚

if ids is not None:
    for i, corner in enumerate(corners):
        # 获取对应标记的 ID
        corner *= resize_scale
        points = corner[0].astype(int)  # 将 corner 转换为整数像素坐标
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=10)

        marker_id = ids[i][0]

        # 定义标记的真实三维坐标（以中心为原点）
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

        # 计算姿态
        retval, rvec, tvec = cv2.solvePnP(obj_points, corner[0], camera_matrix, dist_coeffs)

        # 打印 ID 和外参
        print(f"Marker ID: {marker_id}")
        print(f"Rotation Vector (rvec): {rvec.T}")
        print(f"Translation Vector (tvec): {tvec.T}\n")

        # 在图像上绘制坐标轴
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.02, thickness=20)

        # 在标记中心绘制 ID
        center = tuple(corner[0].mean(axis=0).astype(int))
        cv2.putText(image, f"{marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
                        6, (0, 0, 255), 30, cv2.LINE_AA)

else:
    print("No markers detected.")

# 显示结果
output_path = 'Display_ArUco_Extrinsic.png'
cv2.imwrite(output_path, image)
