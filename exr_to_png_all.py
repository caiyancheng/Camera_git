import pyexr
import os
import numpy as np
import imageio
import glob
from tqdm import tqdm

image_root_path = r"E:\sony_pictures\Color_Rects_Display_G1_whole_process_example"
exr_images = glob.glob(os.path.join(image_root_path, '*.exr'))
for exr_file_name in tqdm(exr_images):
    exr = pyexr.open(exr_file_name)
    exr_data = exr.get()
    normalized_data = (exr_data / 600.0 * 255).clip(0, 255).astype(np.uint8)
    png_file_name = os.path.splitext(exr_file_name)[0] + ".png"
    imageio.imwrite(png_file_name, normalized_data)
