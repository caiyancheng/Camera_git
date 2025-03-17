import os
import HDRutils
# from Vignetting.generate_vignetting_map_P2D import generate_vignetting_map_P2D
import numpy as np
import json
import cv2

image_root_path = r'E:\sony_pictures\Color_Rects_Display_G1'
image_save_path = r'E:\sony_pictures\Color_Rects_Display_G1_whole_process_example'
os.makedirs(image_save_path, exist_ok=True)
files = ['DSC00004_PSMS.ARQ', 'DSC00008_PSMS.ARQ', 'DSC00012_PSMS.ARQ']

# files = ['DSC00032_PSMS.ARQ', 'DSC00036_PSMS.ARQ', 'DSC00040_PSMS.ARQ', 'DSC00044_PSMS.ARQ']
files = [os.path.join(image_root_path, i) for i in files]# RAW input files
# # Merge
# HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True)[0]
# HDRutils.imwrite(os.path.join(image_save_path, 'color_rects_0_4_8_12_merge_ARQ.exr'), HDR_img)
#
# # # Merge + MTF
# HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True, mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
# HDRutils.imwrite(os.path.join(image_save_path, 'color_rects_0_4_8_12_merge_MTF_ARQ.exr'), HDR_img)

# Merge + MTF + Vignetting
# vignetting_scaler_RGB = generate_vignetting_map_P2D(S=4, root_path='Vignetting') #[H,W,3]
vignetting_scaler_RGB = np.load('Vignetting/vignetting_scaler_RGB_S_4.npz')['arr_0']
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True, mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
HDR_img_V = HDR_img / vignetting_scaler_RGB
HDRutils.imwrite(os.path.join(image_save_path, 'color_rects_4_8_12_merge_MTF_vignetting_ARQ.exr'), HDR_img_V)

# Merge + MTF + Vignetting + Undistortion
camera_name = 'a7R3'
with open(f'Camera_Calibration/calibration_result_SONY_{camera_name}_100cm_6_refine_2.json', 'r') as fp:
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

# vignetting_scaler_RGB = generate_vignetting_map_P2D(S=4, root_path='Vignetting')
vignetting_scaler_RGB = np.load('Vignetting/vignetting_scaler_RGB_S_4.npz')['arr_0']
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True, mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
HDR_img_V = HDR_img / vignetting_scaler_RGB
# white_mask = np.ones_like(HDR_img_V)
h, w = HDR_img_V.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
dst = cv2.undistort(HDR_img_V, camera_matrix, dist_coeffs, None, newcameramtx)
# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1)
# dst = cv2.remap(HDR_img_V, mapx, mapy, interpolation=cv2.INTER_LINEAR) #cv2.INTER_LANCZOS4
# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1)
# highres_dst = cv2.remap(HDR_img_V, mapx, mapy, interpolation=cv2.INTER_LANCZOS4)
# dst = cv2.resize(highres_dst, (w, h), interpolation=cv2.INTER_AREA)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
# dst_white_mask = cv2.remap(white_mask, mapx, mapy, interpolation=cv2.INTER_CUBIC) #cv2.INTER_LANCZOS4
# dst_white_mask = dst_white_mask[y:y + h, x:x + w]
# dst = dst / dst_white_mask
# HDRutils.imwrite(os.path.join(image_save_path, '0_4_8_12_center.exr'), dst)
HDRutils.imwrite(os.path.join(image_save_path, 'color_rects_0_4_8_12_merge_MTF_vignetting_undistortion_original.exr'), dst)
# HDRutils.imwrite(os.path.join(image_save_path, 'color_rects_0_4_8_12_merge_MTF_vignetting_undistortion_LINEAR.exr'), dst)