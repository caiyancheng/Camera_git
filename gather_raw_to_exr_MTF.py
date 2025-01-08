import os
import HDRutils
image_root_path = r'E:\sony_pictures\MTF_star_multi_exposure_80cm'
files = ['DSC00000.ARW', 'DSC00001.ARW', 'DSC00002.ARW', 'DSC00003.ARW']
files = [os.path.join(image_root_path, i) for i in files]# RAW input files
HDR_img = HDRutils.merge(files, demosaic_first=False, mtf_json='MTF/mtf_sony_a7R_FE_28_90_new.json')[0]
HDRutils.imwrite(os.path.join(image_root_path, 'merged_0123_MTF.exr'), HDR_img)