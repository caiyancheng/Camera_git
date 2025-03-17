import numpy as np
import HDRutils
import os
import glob
from tqdm import tqdm

image_READ_root_path = r'E:\sony_pictures\pixel_format/'
image_SAVE_root_path = r'E:\sony_pictures\pixel_format_crop/'
if os.path.exists(image_SAVE_root_path) is False:
    os.makedirs(image_SAVE_root_path, exist_ok=True)
images = glob.glob(os.path.join(image_READ_root_path, 'DSC*.ARQ'))
crop_size = (500, 500)
for image in tqdm(images):
    img_data = HDRutils.imread(image)
    center_x, center_y = img_data.shape[1] // 2, img_data.shape[0] // 2
    x_start = center_x - crop_size[1] // 2
    x_end = center_x + crop_size[1] // 2
    y_start = center_y - crop_size[0] // 2
    y_end = center_y + crop_size[0] // 2
    cropped_image = img_data[y_start:y_end, x_start:x_end, :]
    HDRutils.imwrite(os.path.join(image_SAVE_root_path, image.split('\\')[-1].replace('.ARQ', '.png')), cropped_image)