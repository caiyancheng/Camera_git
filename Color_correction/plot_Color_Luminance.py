import numpy as np
import matplotlib.pyplot as plt
import json
import os

sizes_list = [[250,250], [500,500], [1000,1000], [2000,2000], [3840,2160]]
root_path = r'Color_Luminance_Measure/pixel_num_20_repeat_5'

plt.figure(figsize=(8,8))
max_luminance = 0
gamma = 2.2
for sizes in sizes_list:
    with open(os.path.join(root_path, f'color_luminance_measurements_rect_width_{sizes[0]}_rect_height_{sizes[1]}.json'), 'r') as f:
        json_data = json.load(f)
    pixel_value_list = np.array(json_data['pixel_value_list']) / 255
    Luminance_list = json_data['Luminance_list']
    max_luminance = max(max_luminance, max(Luminance_list))
    plt.plot(pixel_value_list, Luminance_list, label=f'Luminance W={sizes[0]} H={sizes[1]}')
L_gamma = max_luminance * (pixel_value_list) ** gamma
plt.plot(pixel_value_list, L_gamma, '--', label=f'Gamma = 2.2')
plt.xlabel('Color')
# plt.ylabel('log10(Luminance)')
plt.ylabel('Luminance')
plt.tight_layout()
plt.legend()
plt.show()
