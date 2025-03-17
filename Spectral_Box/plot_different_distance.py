import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

color_list = ['red', 'green', 'blue', 'white']
color_plot_list = ['red', 'green', 'blue', 'black']

for color_index in tqdm(range(len(color_list))):
    plt.figure(figsize=(6, 6))
    # if color_index != 3:
    #     continue
    color = color_list[color_index]
    json_file_name_near = f'log_json/G1_{color}_255_repeat_10_theta_0.json'
    json_file_name_far = f'log_json/G1_{color}_255_repeat_10_theta_0_far.json'
    with open(json_file_name_near, 'r') as f:
        data = json.load(f)
        lmb_mean_near = np.array(data['lmb_mean'])
        L_mean_near = np.array(data['L_mean'])
        # plt.plot(lmb_mean, L_mean, color=color_plot_list[color_index], linestyle='-')
    with open(json_file_name_far, 'r') as f:
        data = json.load(f)
        lmb_mean_far = np.array(data['lmb_mean'])
        L_mean_far = np.array(data['L_mean'])
    plt.plot(lmb_mean_near, L_mean_far / L_mean_near, color=color_plot_list[color_index])
    plt.axhline(y=1, color='r', linestyle='--', linewidth=1, label="Y = 1")
    plt.xlabel('Wavelength $\lambda$ (nm)')
    plt.ylabel('Power Far / Power Near')
    # plt.yscale('log')
    plt.xlim([380, 700])
    plt.ylim([0.9, 1.1])
    plt.show()


