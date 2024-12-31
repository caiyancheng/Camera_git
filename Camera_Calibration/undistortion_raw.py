import HDRutils
import os
import numpy as np
import json
import cv2 as cv

camera_name = 'a7R1'
with open(f'calibration_result_SONY_{camera_name}.json', 'r') as fp:
    camera_calibration_result = json.load(fp)

mtx = np.array(camera_calibration_result['mtx'])
dist = np.array(camera_calibration_result['dist'])

image_read_path = r'E:\sony_pictures\ArUco_Petterns_JPEG_3/'
image_save_path = r'E:\sony_pictures\ArUco_Petterns_JPEG_3_undistortion/'
os.makedirs(image_save_path, exist_ok=True)

img_name = 'DSC00001.arw'
img = HDRutils.imread(os.path.join(image_read_path, img_name)) #[H,W,3] #[4920,7362,3], 16位, 0 - 65535
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# X = 1
# # crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
HDRutils.imwrite(os.path.join(image_save_path, img_name + 'undistortion.exr'), dst)