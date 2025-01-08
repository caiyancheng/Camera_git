import HDRutils
import os
import glob
from tqdm import tqdm

image_root_path = r'E:\sony_pictures\ArUco_Homo_1/'
images = glob.glob(os.path.join(image_root_path, 'DSC*.ARW'))
for image in tqdm(images):
    img_data = HDRutils.imread(image)
    HDRutils.imwrite(os.path.join(image_root_path, image.replace('ARW', 'png')), img_data)