import numpy as np
import matplotlib.pyplot as plt
import json
import os

sizes_list = [[250,250], [500,500], [1000,1000], [2000,2000], [3840,2160]]
root_path = r'Color_Luminance_Measure/pixel_num_20_repeat_5'

fit_degree = 4
json_save = {}
plt.figure(figsize=(6,6))
for sizes in sizes_list:
    with open(os.path.join(root_path, f'color_luminance_measurements_rect_width_{sizes[0]}_rect_height_{sizes[1]}.json'), 'r') as f:
        json_data = json.load(f)
    pixel_value_list = np.array(json_data['pixel_value_list']) / 255
    log10_Luminance_list = np.log10(np.array(json_data['Luminance_list']))
    plt.scatter(log10_Luminance_list, pixel_value_list, label=f'Luminance W={sizes[0]} H={sizes[1]}')
    x_fit = np.linspace(np.min(log10_Luminance_list), np.max(log10_Luminance_list), 100)
    coefficients = np.polyfit(log10_Luminance_list, pixel_value_list, deg=fit_degree)
    fitted_curve = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, fitted_curve)
    json_save[f'size_w_{sizes[0]}_h_{sizes[1]}'] = {'coefficients': coefficients.tolist(),
                                                    'min_Luminance': np.min(log10_Luminance_list),
                                                    'max_Luminance': np.max(log10_Luminance_list)}
with open(f'fit_L_C_parameters/log10_Luminance_Color_Fit_result_poly_{fit_degree}.json', 'w') as fp:
    json.dump(json_save, fp)
plt.title(f'{fit_degree} degree Polynomial Fit for All Sizes')
plt.ylabel('Color')
# plt.ylabel('log10(Luminance)')
plt.xlabel('Log10 Luminance')
plt.tight_layout()
plt.legend()
plt.show()
