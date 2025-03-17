import HDRutils
import os
import glob
from tqdm import tqdm

image_root_path = r'E:\sony_pictures\MTF_star_a7R3_100cm'
# image_root_path = r'E:\sony_pictures\a7R3_100_aruco_4_demosaic_100_distance'
# images = glob.glob(os.path.join(image_root_path, 'DSC*.ARW'))
images = glob.glob(os.path.join(image_root_path, 'DSC*.ARQ'))
for image in tqdm(images):
    img_data = HDRutils.imread(image, color_space='raw')
    # HDRutils.imwrite(os.path.join(image_root_path, image.replace('ARW', 'png')), img_data)
    HDRutils.imwrite(os.path.join(image_root_path, image.replace('ARQ', 'png')), img_data)