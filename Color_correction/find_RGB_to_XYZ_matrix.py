import numpy as np
import json

def compute_rgb2xyz_matrix(RGBs, XYZs):
    M, residuals, _, _ = np.linalg.lstsq(RGBs, XYZs, rcond=None)
    return M.T, residuals

def xyY_to_XYZ(xyY):
    x, y, Y = xyY
    if y == 0:  # 避免除以零
        return np.array([0, Y, 0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])


def expand_rgb_features(rgb_array):
    R, G, B = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]
    RG_sqrt = np.sqrt(R * G)
    RB_sqrt = np.sqrt(R * B)
    GB_sqrt = np.sqrt(G * B)

    return np.column_stack((R, G, B, RG_sqrt, RB_sqrt, GB_sqrt))

with open('Camera_Colorchecker_MeanRGB.json', 'r') as f:
    data_Camera_Colorchecker_MeanRGB = json.load(f)
with open('Color_measure_specbos.json', 'r') as f:
    data_Color_measure_specbos = json.load(f)


keys_list = list(data_Camera_Colorchecker_MeanRGB.keys())
color_num = 24
RGB_camera_linear = np.zeros([color_num, 3])
XYZ_specbos_linear = np.zeros([color_num, 3])
for key_index in range(len(keys_list)):
    key_name = keys_list[key_index]
    exr_file_name = key_name.split('\\')[-1].split('.')[0]
    RGB_camera_linear[key_index,:] = np.array(data_Camera_Colorchecker_MeanRGB[key_name])
    specbos_measure = data_Color_measure_specbos[exr_file_name]
    specbos_xyY_linear = np.array([specbos_measure['x_mean'], specbos_measure['y_mean'], specbos_measure['Y_mean']])
    XYZ_specbos_linear[key_index,:] = xyY_to_XYZ(specbos_xyY_linear)
RGB_camera_linear_expanded = expand_rgb_features(RGB_camera_linear)
json_data_dict_1 = {'RGB_camera_linear': RGB_camera_linear.tolist(), 'XYZ_specbos_linear': XYZ_specbos_linear.tolist()}
with open('RGB_camera_XYZ_specbos_linear.json', 'w') as f:
    json.dump(json_data_dict_1, f)

Matrix_RGB2XYZ, RGB2XYZ_residuals = compute_rgb2xyz_matrix(RGB_camera_linear, XYZ_specbos_linear)
# RGB2XYZ_residuals = [153.54755438 276.69926624 296.67120636]
Matrix_RGB_expanded_2XYZ, RGB_expanded_2XYZ_residuals = compute_rgb2xyz_matrix(RGB_camera_linear_expanded, XYZ_specbos_linear)
# RGB_expanded_2XYZ_residuals = [149.46341814 245.09325021 274.50890376]
json_data_dict = {'Matrix_RGB2XYZ': Matrix_RGB2XYZ.tolist(),
                  'Matrix_RGB_expanded_2XYZ': Matrix_RGB_expanded_2XYZ.tolist()}
with open('Camera_Colorchecker_RGB2XYZ.json', 'w') as f:
    json.dump(json_data_dict, f)