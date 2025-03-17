import numpy as np
import matplotlib.pyplot as plt
import json
import os

sizes_list = [[250,250], [500,500], [1000,1000], [2000,2000], [3840,2160]]
root_path = r'Color_Luminance_Measure/pixel_num_20_repeat_5'

fit_degree = 7
json_save = {}
plt.figure(figsize=(6,6))
max_luminance = 0
for sizes in sizes_list:
    with open(os.path.join(root_path, f'color_luminance_measurements_rect_width_{sizes[0]}_rect_height_{sizes[1]}.json'), 'r') as f:
        json_data = json.load(f)
    pixel_value_list = np.array(json_data['pixel_value_list'])
    Luminance_list = np.array(json_data['Luminance_list'])
    max_luminance = max(max_luminance, max(Luminance_list))
    plt.scatter(pixel_value_list, Luminance_list, label=f'Luminance W={sizes[0]} H={sizes[1]}')
    x_fit = np.linspace(np.min(pixel_value_list), np.max(pixel_value_list), 100)
    coefficients = np.polyfit(pixel_value_list, Luminance_list, deg=fit_degree)
    fitted_curve = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, fitted_curve)
    json_save[f'size_w_{sizes[0]}_h_{sizes[1]}'] = {'coefficients': coefficients.tolist()}
with open(f'fit_L_C_parameters/Color_Luminance_Fit_result_poly_{fit_degree}.json', 'w') as fp:
    json.dump(json_save, fp)
L_gamma_2_2 = max_luminance * (pixel_value_list / 255) ** 2.2
L_gamma_2_4 = max_luminance * (pixel_value_list / 255) ** 2.4
L_gamma_2_6 = max_luminance * (pixel_value_list / 255) ** 2.6
plt.plot(pixel_value_list, L_gamma_2_2, '--', label=f'Gamma = 2.2')
plt.plot(pixel_value_list, L_gamma_2_4, '--', label=f'Gamma = 2.4')
plt.plot(pixel_value_list, L_gamma_2_6, '--', label=f'Gamma = 2.6')
plt.title(f'{fit_degree} degree Polynomial Fit for All Sizes')
plt.xlabel('Color')
# plt.ylabel('log10(Luminance)')
plt.ylabel('Luminance')
plt.tight_layout()
plt.legend()
plt.show()
