import numpy as np
import json
import cv2 as cv

camera_name = 'a7R1'
with open(f'calibration_result_SONY_{camera_name}.json', 'r') as fp:
    camera_calibration_result = json.load(fp)

mtx = np.array(camera_calibration_result['mtx'])
dist = np.array(camera_calibration_result['dist'])

image_path = r'E:\sony_pictures\ArUco_Petterns_JPEG_3/DSC00000.JPG'

img = cv.imread(image_path)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_alpha_1.png', dst)