import pyexr
import os
import numpy as np
import imageio
import glob
from tqdm import tqdm
from Color_space_Transform import cm_xyz2rgb
from RGB_2_XYZ_color_correction import transform_RGB_2_XYZ_color_correction
from display_encoding import display_encode
display_encode_tool = display_encode(500)

image_root_path = r"E:\sony_pictures\Color_Fringing_2025_2_23_whole_process_new"
exr_images = ['CC_XYZ_Homo_Transform_aruco_MTF_vignetting_undistortion_remap_INTER_LANCZOS4_center_crop_undistortion_remap_5.exr']
exr_images = [os.path.join(image_root_path, i) for i in exr_images]
for exr_file_name in tqdm(exr_images):
    exr = pyexr.open(exr_file_name)
    exr_data_camera_linear = exr.get()
    exr_data_camera_linear[exr_data_camera_linear < 0] = 0
    exr_data_XYZ_linear = transform_RGB_2_XYZ_color_correction(exr_data_camera_linear, expand=True)
    exr_data_XYZ_linear[exr_data_XYZ_linear < 0] = 0
    exr_data_RGB_linear = cm_xyz2rgb(exr_data_XYZ_linear, 'sRGB')
    exr_data_RGB_linear[exr_data_RGB_linear < 0] = 0
    # exr_data_RGB_encoded = display_encode_tool.L2C_sRGB(exr_data_RGB_linear)
    exr_data_RGB_encoded = (exr_data_RGB_linear / 400) ** (1/2.2)
    exr_data_RGB_clipped = np.clip(exr_data_RGB_encoded, 0, 1) * 255
    exr_data_uint8 = exr_data_RGB_clipped.round().astype(np.uint8)
    png_file_name = os.path.splitext(exr_file_name)[0] + "_color_correction.png"
    imageio.imwrite(png_file_name, exr_data_uint8)
