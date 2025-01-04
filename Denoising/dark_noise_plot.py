import numpy as np
import matplotlib.pyplot as plt
import os
import HDRutils
import json
from tqdm import tqdm
read_and_write = False
plot = True

image_root_path = 'E:\sony_pictures\Dark_Pictures_2'
image_num_list = list(range(0,24))
image_list = [f"DSC{num:05d}.ARW" for num in image_num_list]
exposure_time_list = [1/20, 1/20, 1/10, 1/10, 1/8, 1/8, 1/4, 1/4,
                      1/2, 1/2, 1, 1, 2, 2, 4, 4, 8, 8, 10, 10,
                      20, 20, 30, 30]

if read_and_write:
    raw_mean_list = []
    for image_index in tqdm(range(len(image_list))):
        image_path = os.path.join(image_root_path, image_list[image_index])
        exposure_time = exposure_time_list[image_index]
        image_data = HDRutils.imread(image_path)
        raw_mean = image_data.mean()
        raw_mean_list.append(raw_mean)
    json_data = {
        'exposure_time_list': exposure_time_list,
        'raw_mean_list': raw_mean_list,
    }
    with open('dark_noise_data.json', 'w') as f:
        json.dump(json_data, f)
else:
    with open('dark_noise_data.json', 'r') as f:
        dark_noise_data = json.load(f)
    exposure_time_list = dark_noise_data['exposure_time_list']
    raw_mean_list = dark_noise_data['raw_mean_list']


if plot:
    plt.xlabel('Exposure Time (s)')
    plt.ylabel('Raw Image Mean Value')
    plt.xscale('log')
    plt.plot(exposure_time_list, raw_mean_list, 'b')
    plt.scatter(exposure_time_list, raw_mean_list, marker='o', color='red', label='Measurements')
    plt.show()

