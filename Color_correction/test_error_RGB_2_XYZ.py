from RGB_2_XYZ_color_correction import transform_RGB_2_XYZ_color_correction
import numpy as np
import json

with open('RGB_camera_XYZ_specbos_linear.json', 'r') as f:
    data_RGB_camera_XYZ_specbos_linear = json.load(f)

RGB_camera_linear = np.array(data_RGB_camera_XYZ_specbos_linear['RGB_camera_linear'])
XYZ_specbos_linear = np.array(data_RGB_camera_XYZ_specbos_linear['XYZ_specbos_linear'])

rmse_loss_normal = []
rmse_loss_expand = []
percentage_error_normal = []
percentage_error_expand = []

for RGB_index in range(len(RGB_camera_linear)):
    RGBs = RGB_camera_linear[RGB_index]
    XYZs = transform_RGB_2_XYZ_color_correction(RGBs, expand=False)
    XYZs_expand = transform_RGB_2_XYZ_color_correction(RGBs, expand=True)
    XYZs_gt = XYZ_specbos_linear[RGB_index]

    rmse_normal = np.sqrt(np.mean((XYZs - XYZs_gt) ** 2))
    rmse_expand = np.sqrt(np.mean((XYZs_expand - XYZs_gt) ** 2))
    rmse_loss_normal.append(rmse_normal)
    rmse_loss_expand.append(rmse_expand)

    percentage_error_normal.append(np.mean(np.abs((XYZs - XYZs_gt) / XYZs_gt)) * 100)
    percentage_error_expand.append(np.mean(np.abs((XYZs_expand - XYZs_gt) / XYZs_gt)) * 100)

mean_rmse_loss_normal = np.mean(rmse_loss_normal)
mean_rmse_loss_expand = np.mean(rmse_loss_expand)
mean_percentage_error_normal = np.mean(percentage_error_normal)
mean_percentage_error_expand = np.mean(percentage_error_expand)

print("Mean RMSE Loss (XYZs vs XYZs_gt):", mean_rmse_loss_normal)
print("Mean RMSE Loss (XYZs_expand vs XYZs_gt):", mean_rmse_loss_expand)
print("Mean Percentage Error (XYZs vs XYZs_gt):", mean_percentage_error_normal, "%")
print("Mean Percentage Error (XYZs_expand vs XYZs_gt):", mean_percentage_error_expand, "%")



