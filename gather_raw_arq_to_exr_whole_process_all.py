import os
import HDRutils
from Vignetting.generate_vignetting_map_P2D import generate_vignetting_map_P2D
import numpy as np
import json
import cv2
from tqdm import tqdm

image_root_path = r'E:\sony_pictures\Color_Fringing_2025_2_23'
image_save_path = r'E:\sony_pictures\Color_Fringing_2025_2_23_whole_process'
os.makedirs(image_save_path, exist_ok=True)
files_list = [['DSC00000_PSMS.ARQ', 'DSC00004_PSMS.ARQ'],['DSC00008_PSMS.ARQ', 'DSC00012_PSMS.ARQ'],
              ['DSC00016_PSMS.ARQ', 'DSC00020_PSMS.ARQ'],['DSC00024_PSMS.ARQ', 'DSC00028_PSMS.ARQ']]

save_exr_name_list = ['aruco.exr', 'color_fringing_letter.exr', 'color_fringing_white_circle.exr', 'color_fringing_black_circle.exr']

camera_name = 'a7R3'
with open(f'Camera_Calibration/calibration_result_SONY_{camera_name}_100cm_6_refine.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

vignetting_scaler_RGB = generate_vignetting_map_P2D(S=4, root_path='Vignetting')

for gather_index in tqdm(range(len(files_list))):
    files = files_list[gather_index]
    save_exr_name = save_exr_name_list[gather_index]
    files = [os.path.join(image_root_path, i) for i in files]  # RAW input files

    HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True,
                             mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
    HDR_img_V = HDR_img / vignetting_scaler_RGB
    h, w = HDR_img_V.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(HDR_img_V, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    HDRutils.imwrite(os.path.join(image_save_path, save_exr_name), dst)