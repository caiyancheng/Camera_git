import os
import HDRutils
image_root_path = r'E:\sony_pictures\MTF_star_a7R3_100cm'
files = ['DSC00000.ARW', 'DSC00004.ARW']
files = [os.path.join(image_root_path, i) for i in files]# RAW input files
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', mtf_json='MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
HDRutils.imwrite(os.path.join(image_root_path, 'merged_04_ARW_mtf.exr'), HDR_img)