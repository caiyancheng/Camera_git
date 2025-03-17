import numpy as np
import json

with open(r'E:\Py_codes\Camera_git\Color_correction/Camera_Colorchecker_RGB2XYZ.json') as json_file:
    data = json.load(json_file)
Matrix_RGB2XYZ = np.array(data['Matrix_RGB2XYZ'])
Matrix_RGB_expanded_2XYZ = np.array(data['Matrix_RGB_expanded_2XYZ'])

def expand_rgb_features(rgb_array):
    R, G, B = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    RG_sqrt = np.sqrt(R * G)
    RB_sqrt = np.sqrt(R * B)
    GB_sqrt = np.sqrt(G * B)
    return np.stack([R, G, B, RG_sqrt, RB_sqrt, GB_sqrt], axis=-1)

def transform_RGB_2_XYZ_color_correction(RGBs, expand=True):
    if expand:
        RGBs = expand_rgb_features(RGBs)
        XYZs = RGBs @ Matrix_RGB_expanded_2XYZ.T
    else:
        XYZs = RGBs @ Matrix_RGB2XYZ.T
    return XYZs

if __name__ == '__main__':
    RGBs = np.array([[160.23, 416.42, 271.01]]) #np.array([[74.27, 163.38, 105.95]])
    XYZs = transform_RGB_2_XYZ_color_correction(RGBs, expand=False)
    print(XYZs)
    XYZs = transform_RGB_2_XYZ_color_correction(RGBs, expand=True)
    print(XYZs)


