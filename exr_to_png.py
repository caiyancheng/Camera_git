import pyexr
import os
import numpy as np
import imageio

exr_file_name = r"E:\sony_pictures\Color_Rects_Display_G1_whole_process_example/color_rects_0_4_8_12_merge_MTF_vignetting_ARQ.exr"
exr = pyexr.open(exr_file_name)
exr_data = exr.get()
normalized_data = (exr_data / 1000.0 * 255).clip(0, 255).astype(np.uint8)
png_file_name = os.path.splitext(exr_file_name)[0] + ".png"
imageio.imwrite(png_file_name, normalized_data)
