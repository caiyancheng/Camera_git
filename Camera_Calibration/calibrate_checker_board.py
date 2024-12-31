import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm
import os
import json
# display_width_meter = 1.225
# # display_height_meter = 0.706
# display_width_pixels = 3840
# # display_height_pixels = 2160
# m_p_w = display_width_meter / display_width_pixels
# # m_p_h = display_height_meter / display_height_pixels
square_size = 0.01 #in meters
# square_size_m = square_size * m_p_w

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
checkerboard_size_row = 18 #9
checkerboard_size_col = 27 #14
x, y = np.mgrid[0:checkerboard_size_col, 0:checkerboard_size_row]
objp = np.zeros((checkerboard_size_row * checkerboard_size_col, 3), np.float32)
objp[:, :2] = np.column_stack((x.ravel(), y.ravel()))
objp[:,0] *= square_size
objp[:,1] *= square_size

center_x = (np.max(objp[:, 0]) + np.min(objp[:, 0])) / 2
center_y = (np.max(objp[:, 1]) + np.min(objp[:, 1])) / 2
objp[:, 0] -= center_x
objp[:, 1] -= center_y

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

image_root_path = r'E:\sony_pictures\checker_board_pictures_cm_E05_F8_ISO_100_v1'
save_clibrate_image_path = image_root_path + '_calibrate'
os.makedirs(save_clibrate_image_path, exist_ok=True)
images = glob.glob(os.path.join(image_root_path, 'DSC*.jpg'))
# resize_scale_list = np.logspace(np.log10(2), np.log10(10), 20)
resize_scale_list = [1]

save_images = False
for fname in tqdm(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for resize_scale in resize_scale_list:
        new_height = round(img.shape[0] / resize_scale)
        new_width = round(img.shape[1] / resize_scale)
        if resize_scale == 1:
            gray_resize = gray.copy()
        else:
            gray_resize = cv.resize(gray, (new_width, new_height))

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray_resize, (checkerboard_size_row, checkerboard_size_col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners *= resize_scale
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            if save_images:
                cv.drawChessboardCorners(img, (checkerboard_size_row, checkerboard_size_col), corners2, ret)
                cv.imwrite(os.path.join(save_clibrate_image_path, fname.split('\\')[-1]), img)
        else:
            print('Cannot find the checker board!')

# cv.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


json_data = {'ret': ret, 'mtx': mtx.tolist(), 'dist': dist.tolist()}
with open('calibration_result_SONY_a7R1.json', 'w') as outfile:
    json.dump(json_data, outfile)