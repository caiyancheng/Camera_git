import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

color_list = ['red', 'green', 'blue', 'white']
color_pure_value_list = [32, 64, 128, 255]
color_plot_list = ['red', 'green', 'blue', 'black']
plt.figure(figsize=(6,6))
for color_index in tqdm(range(len(color_list))):
    if color_index != 3:
        continue
    color = color_list[color_index]
    for color_pure_value in color_pure_value_list:
        json_file_name = f'G1_{color}_{color_pure_value}_repeat_10.json'
        with open(json_file_name, 'r') as f:
            data = json.load(f)
            lmb_mean = data['lmb_mean']
            L_mean = data['L_mean']
            plt.plot(lmb_mean, L_mean, color=color_plot_list[color_index])
plt.xlabel('Wavelength $\lambda$ (nm)')
plt.ylabel('Power')
plt.yscale('log')
plt.ylim([0.00001, 0.01])
plt.show()


