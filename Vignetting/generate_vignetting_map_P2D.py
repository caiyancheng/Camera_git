import os
import json
import numpy as np
import scipy.io as sio
from PIL import Image


# **解析多项式模型并计算拟合值**
def polyval2d(model, x, y):
    # 解析 ModelTerms 和 Coefficients
    model_terms = model['ModelTerms'][0, 0]  # 取出 ModelTerms 数组
    coefficients = model['Coefficients'][0, 0].flatten()  # 取出 Coefficients

    # 计算多项式值
    V_fit = np.zeros_like(x, dtype=np.float64)
    for i, (px, py) in enumerate(model_terms):
        V_fit += coefficients[i] * (x ** px) * (y ** py)

    return V_fit


def generate_vignetting_map_P2D(S=2, H=5320, W=7968, root_path=None):
    # **加载 .mat 文件**
    param_file = f'bivariate_polyfit_S{S}_P2D.mat'  # 你的 .mat 文件名
    if root_path is not None:
        param_file = os.path.join(root_path, param_file)
    mat_data = sio.loadmat(param_file)
    # 提取存储的模型
    models_RGB = mat_data['models_RGB']
    # **设定图像大小**
    x_grid, y_grid = np.meshgrid(np.arange(1, W + 1), np.arange(1, H + 1))
    x_flat, y_flat = x_grid.ravel(), y_grid.ravel()
    # **计算所有像素点的值**
    vignetting_scaler_RGB = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):  # R/G/B
        model = models_RGB[0, c]  # 取出对应通道的模型
        V_fit = polyval2d(model, x_flat, y_flat)  # 计算拟合值
        V_fit_normalize = V_fit / V_fit.max()
        vignetting_scaler_RGB[:, :, c] = V_fit_normalize.reshape(H, W)

    return vignetting_scaler_RGB


def save_vignetting_maps(vignetting_scaler_RGB, save_prefix="vignetting"):
    channels = ['R', 'G', 'B']

    for i, channel in enumerate(channels):
        # 取出单通道并转换到 0-255 范围
        channel_data = (vignetting_scaler_RGB[:, :, i] * 255).astype(np.uint8)

        # 创建 PIL Image 并保存
        img = Image.fromarray(channel_data, mode='L')
        img.save(f"{save_prefix}_{channel}.png")
        print(f"Saved {save_prefix}_{channel}.png")


if __name__ == '__main__':
    vignetting_scaler_RGB = generate_vignetting_map_P2D(S=4)
    vignetting_scaler_RGB_float16 = vignetting_scaler_RGB.astype(np.float16)
    # save_vignetting_maps(vignetting_scaler_RGB)
    # with open('vignetting_scaler_RGB_S_4.json', 'w') as f:
    #     json.dump(vignetting_scaler_RGB, f)
    np.savez_compressed('vignetting_scaler_RGB_S_4.npz', vignetting_scaler_RGB_float16)
