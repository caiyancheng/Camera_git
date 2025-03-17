import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

color_list = ['red', 'green', 'blue', 'white']
color_plot_list = ['red', 'green', 'blue', 'black']
angle_list = [0, 45, 80]
line_style_list = ['-', '--', '-.']

for color_index in tqdm(range(len(color_list))):
    plt.figure(figsize=(6, 6))
    # if color_index != 3:
    #     continue
    color = color_list[color_index]
    json_file_name_base = f'log_json/G1_{color}_255_repeat_10_theta_0.json'
    with open(json_file_name_base, 'r') as f:
        data = json.load(f)
        lmb_mean_base = np.array(data['lmb_mean'])
        L_mean_base = np.array(data['L_mean'])

    for angle_index in range(len(angle_list)):
        angle = angle_list[angle_index]
        json_file_name = f'log_json/G1_{color}_255_repeat_10_theta_{angle}.json'
        with open(json_file_name, 'r') as f:
            data = json.load(f)
            lmb_mean = np.array(data['lmb_mean'])
            L_mean = np.array(data['L_mean'])
        plt.plot(lmb_mean, L_mean / L_mean_base, color=color_plot_list[color_index], linestyle=line_style_list[angle_index])
    plt.xlabel('Wavelength $\lambda$ (nm)')
    plt.axhline(y=1, color='r', linestyle='--', linewidth=1, label="Y = 1")
    plt.ylabel('Power Tilt / Power Vertical')
    # plt.yscale('log')
    plt.xlim([380, 780])
    plt.ylim([0.1,1.1])
    plt.tight_layout()
    plt.show()


