import pyexr
import os
import numpy as np
import imageio
import glob
from tqdm import tqdm
import json


json_data_dict = {}
image_root_path = r"E:\sony_pictures\Color_Checker_100cm_whole_process/"
exr_images = glob.glob(os.path.join(image_root_path, '*.exr'))
for exr_file_name in tqdm(exr_images):
    exr = pyexr.open(exr_file_name)
    exr_data = exr.get()
    mean_RGB = np.mean(exr_data, axis=(0, 1))
    exr_file_name_key = exr_file_name.split('\\')[-1].split('.')[0]
    json_data_dict[exr_file_name] = mean_RGB.tolist()

with open('Camera_Colorchecker_MeanRGB.json', 'w') as outfile:
    json.dump(json_data_dict, outfile)

