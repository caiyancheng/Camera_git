import os
import HDRutils
image_root_path = r'E:\sony_pictures\Vignetting_2'
image_save_path = r'E:\sony_pictures\Vignetting_2_merge'
os.makedirs(image_save_path, exist_ok=True)
files = ['DSC00048_PSMS.ARQ', 'DSC00052_PSMS.ARQ', 'DSC00056_PSMS.ARQ']
files = [os.path.join(image_root_path, i) for i in files]# RAW input files
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True, mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
HDRutils.imwrite(os.path.join(image_save_path, '48_52_56_merge_focus_distance_100cm_real_distance_20cm.exr'), HDR_img)