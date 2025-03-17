import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Color_space_Transform import cm_xyz2rgb

color = 'white' #'red', 'green', 'blue', 'white'
angle = 0#0, 45, 80
print(f'color - {color}; angle - {angle}')
SPD_json_file = f'log_json/G1_{color}_255_repeat_10_theta_{angle}.json'
with open(SPD_json_file, 'r') as f:
    SPD_data = json.load(f)
lmb_mean = np.array(SPD_data['lmb_mean'])
L_mean = np.array(SPD_data['L_mean'])
CIE_file = '../CIE_xyz_1931_2deg.csv'
CIE_data = pd.read_csv(CIE_file, header=None)
CIE_wavelength_array = np.array(list(CIE_data[0]))
CIE_X_array = np.array(list(CIE_data[1]))
CIE_Y_array = np.array(list(CIE_data[2]))
CIE_Z_array = np.array(list(CIE_data[3]))

indices = np.array([np.where(lmb_mean == lamda)[0][0] for lamda in CIE_wavelength_array])
X = np.sum(CIE_X_array * L_mean[indices])
Y = np.sum(CIE_Y_array * L_mean[indices])
Z = np.sum(CIE_Z_array * L_mean[indices])
k = 1 / Y
L = 100
X = k * L * X
Y = k * L * Y
Z = k * L * Z
print(f'X: {X}, Y: {Y}, Z: {Z}')
rgb = cm_xyz2rgb(np.array([X,Y,Z]), 'sRGB')
print(f'R: {rgb[0]}, G: {rgb[1]}, B: {rgb[2]}')