import json
from Color_space_Transform import cm_xyz2rgb
import numpy as np

color = 'white' #'red', 'green', 'blue', 'white'
angle = 0#0, 45, 80
print(f'color - {color}; angle - {angle}')
json_file = f'log_json/G1_{color}_255_repeat_10_theta_{angle}.json'
with open(json_file, 'r') as f:
    json_data = json.load(f)
Y_mean = json_data['Y_mean']
x_mean = json_data['x_mean']
y_mean = json_data['y_mean']

X = x_mean / y_mean * Y_mean
Z = (1 - x_mean - y_mean) / y_mean * Y_mean
xyz = np.array([X, Y_mean, Z])
print(f'xyz - {xyz}')
RGB = cm_xyz2rgb(xyz)
print(f'RGB - {RGB}')