import numpy as np
import HDRutils
import os
import json
import cv2

def photo_png_post_process_MTF_undistortion(png_file_name, camera_matrix_file, mtf_json_file):
    with open(camera_matrix_file, 'r') as fp:
        camera_calibration_result = json.load(fp)
    camera_matrix = np.array(camera_calibration_result['mtx'])
    dist_coeffs = np.array(camera_calibration_result['dist'])
    img = HDRutils.merge(png_file_name, demosaic_first=False, mtf_json=mtf_json_file)[0]
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def photo_png_post_process_undistortion(png_file_name, camera_matrix_file):
    with open(camera_matrix_file, 'r') as fp:
        camera_calibration_result = json.load(fp)
    camera_matrix = np.array(camera_calibration_result['mtx'])
    dist_coeffs = np.array(camera_calibration_result['dist'])
    img = HDRutils.merge(png_file_name, demosaic_first=False, mtf_json=mtf_json_file)[0]
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

if __name__ == "__main__":
    png_file_name = r'E:\sony_pictures\Homographic_1/DSC00002.png'
    mtf_json_file = 'MTF/mtf_sony_a7R_FE_28_90_new.json'
    camera_matrix_file = r'Camera_Calibration/calibration_result_SONY_a7R1_80cm.json'
    photo_png_post_process_undistortion(png_file_name=png_file_name, camera_matrix_file=camera_matrix_file)
