import cv2
import numpy as np
import HDRutils

# 加载图片
image = cv2.imread(r"E:\sony_pictures\Homographic_1/DSC00001.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resize_scale = 4
new_height = round(image.shape[0] / resize_scale)
new_width = round(image.shape[1] / resize_scale)
gray_resize = cv2.resize(gray, (new_width, new_height))

chessboard_size = (19, 19)
ret, corners = cv2.findChessboardCorners(gray_resize, chessboard_size, None)
corners *= resize_scale
if ret:
    # 亚像素优化角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
    cv2.imwrite('Corners_Detect.png', image)
else:
    print("无法检测到棋盘角点，请重新拍摄或调整棋盘参数。")
