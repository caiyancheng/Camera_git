import cv2
import numpy as np
import os

# img_root_path = 'E:\Py_codes\Camera_git\ArUco_Pattern\Photos_sony_a7R1'
img_root_path = 'E:\sony pictures\ArUco_Patterns_JPEG_2'

# 设置Aruco字典和参数
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# 准备标定所需的变量
obj_points = []  # 3D世界坐标点
img_points = []  # 2D图像坐标点

# 你拍摄的带有Aruco标记的图像路径
# images = ["DSC00001.jpg", "DSC00003.jpg", "DSC00005.jpg", "DSC00007.jpg"]  # 请替换为实际路径
images = ["DSC00601.jpg"]

for image_name in images:
    image_path = os.path.join(img_root_path, image_name)
    img = cv2.imread(image_path)
    img_width = img.shape[1]
    img_height = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = 13
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # 如果标记被找到
    if len(corners) > 0:
        # 这里的obj_point是Aruco标记的3D点 (例如标记的四个角坐标)
        # 在Aruco标定中，我们假设标记在世界坐标系中的位置是已知的
        obj_points.append(np.array([
            [-0.5, -0.5, 0],  # 这个是根据标记的尺寸来定义的
            [0.5, -0.5, 0],
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0]
        ], dtype=np.float32))  # 示例，具体的3D点要根据你的实际标定板

        img_points.append(corners[0][0])

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(obj_points, img_points, gray.shape[::-1], None, None)
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 输出标定结果
print("标定结果：")
print(f"相机内参矩阵：\n{mtx}")
print(f"畸变系数：\n{dist}")
print(f"旋转向量：\n{rvecs}")
print(f"平移向量：\n{tvecs}")
