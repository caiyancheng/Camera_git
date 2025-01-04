import os
import HDRutils
image_root_path = r'E:\sony_pictures\MTF_star_multi_exposure_2'
files = ['DSC00000.ARW', 'DSC00001.ARW', 'DSC00002.ARW'] #, 'DSC00003.ARW']
files = [os.path.join(image_root_path, i) for i in files]# RAW input files
HDR_img = HDRutils.merge(files)[0]
HDRutils.imwrite(os.path.join(image_root_path, 'merged_012.exr'), HDR_img)