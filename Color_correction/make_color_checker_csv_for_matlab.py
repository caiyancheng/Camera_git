import pandas as pd
import json
import os
import numpy as np

def xyY_to_XYZ(xyY):
    x, y, Y = xyY
    if y == 0:  # 避免除以零
        return np.array([0, Y, 0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])

with open('Camera_Colorchecker_MeanRGB.json', 'r') as f:
    data_Camera_Colorchecker_MeanRGB = json.load(f)
root_path = os.path.dirname(list(data_Camera_Colorchecker_MeanRGB.keys())[0])
with open('Color_measure_specbos.json', 'r') as f:
    data_Color_measure_specbos = json.load(f)

csv_file_R_dict = {'row':[], 'col':[], 'color_id':[], 'R': [], 'G': [], 'B': []}
csv_file_M_dict = {'row':[], 'col':[], 'color_id':[], 'X': [], 'Y': [], 'Z': []}

row_num = 4
col_num = 6
for row in range(row_num):
    real_row = row + 1
    for col in range(col_num):
        real_col = col + 1
        color_index = row * col_num + col
        color_id = chr(ord('A') + real_col - 1) + str(real_row)
        csv_file_M_dict['row'].append(real_row)
        csv_file_M_dict['col'].append(real_col)
        csv_file_M_dict['color_id'].append(color_id)
        csv_file_R_dict['row'].append(real_row)
        csv_file_R_dict['col'].append(real_col)
        csv_file_R_dict['color_id'].append(color_id)
        key_name_R = os.path.join(root_path, f'C_{color_index}.exr')
        RGBs = data_Camera_Colorchecker_MeanRGB[key_name_R]
        csv_file_R_dict['R'].append(RGBs[0])
        csv_file_R_dict['G'].append(RGBs[1])
        csv_file_R_dict['B'].append(RGBs[2])
        key_name_M = f'C_{color_index}'
        specbos_measure = data_Color_measure_specbos[key_name_M]
        specbos_xyY_linear = np.array([specbos_measure['x_mean'], specbos_measure['y_mean'], specbos_measure['Y_mean']])
        XYZs = xyY_to_XYZ(specbos_xyY_linear)
        csv_file_M_dict['X'].append(XYZs[0])
        csv_file_M_dict['Y'].append(XYZs[1])
        csv_file_M_dict['Z'].append(XYZs[2])

csv_file_R_df = pd.DataFrame(csv_file_R_dict)
csv_file_M_df = pd.DataFrame(csv_file_M_dict)
csv_file_R_df.to_csv('find_color_correction_matlab/sonya7r3_oled_lg_g1_rgb.csv', index=False)
csv_file_M_df.to_csv('find_color_correction_matlab/sonya7r3_oled_lg_g1_xyz.csv', index=False)

