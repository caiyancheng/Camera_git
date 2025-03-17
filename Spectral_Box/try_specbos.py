import numpy as np
from gfxdisp.specbos import specbos_measure, specbos_get_sprad
import matplotlib.pyplot as plt
import json

repeat_times = 10

json_data = {}
Y_list = []
x_list = []
y_list = []
lmb_list = []
L_list = []
for i in range(repeat_times):
    (Y, x, y) = specbos_measure()
    # lmb, L = specbos_get_sprad()
    Y_list.append(Y)
    x_list.append(x)
    y_list.append(y)
    # lmb_list.append(lmb)
    # L_list.append(L)
Y_mean = np.mean(Y_list, axis=0)
x_mean = np.mean(x_list, axis=0)
y_mean = np.mean(y_list, axis=0)
lmb_mean = np.mean(lmb_list, axis=0)
L_mean = np.mean(L_list, axis=0)
json_data = {'Y_list': Y_list, 'x_list': x_list, 'y_list': y_list, 'lmb_list': lmb_list, 'L_list': L_list,
             'Y_mean': Y_mean, 'x_mean': x_mean, 'y_mean': y_mean, 'lmb_mean': lmb_mean.tolist(), 'L_mean': L_mean.tolist(),
             'repeat_times': repeat_times}
plt.plot(lmb_mean, L_mean)
plt.show()

with open(fr'log_json/G1_red_255_repeat_{repeat_times}_theta_80.json', 'w') as f:
    json.dump(json_data, f)
