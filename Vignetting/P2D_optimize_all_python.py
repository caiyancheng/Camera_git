import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pyexr

def polynomial_fit(x, y, z, degree):
    """对给定的 x, y, z 数据进行多项式拟合"""
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(np.column_stack((x, y)))
    model = LinearRegression()
    model.fit(X_poly, z)
    return model, poly


def evaluate_model(model, poly, x, y):
    """计算多项式拟合的值"""
    X_poly = poly.transform(np.column_stack((x, y)))
    return model.predict(X_poly)


def save_to_json(file_path, models_RGB, errors_RGB, max_values):
    """保存模型参数到 JSON 文件"""
    data = {
        "models_RGB": [{
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_
        } for model, _ in models_RGB],
        "errors_RGB": errors_RGB,
        "max_values": max_values
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(file_path):
    """从 JSON 文件加载模型参数"""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    exr_path = r"E:\\sony_pictures\\Vignetting/merged_16_20_ARQ_mtf_focus_distance_100cm_real_distance_40cm.exr"
    S = 3  # 多项式阶数
    param_file = f"bivariate_polyfit_S{S}_P2D.json"

    exr_file = pyexr.open(exr_path)
    img = exr_file.get()
    H, W, C = img.shape

    if C != 3:
        raise ValueError("输入的 EXR 文件必须是三通道 (RGB) 图像")

    if os.path.exists(param_file):
        print(f"找到已有的优化参数文件 ({param_file})，直接加载...")
        data = load_from_json(param_file)
        models_RGB, errors_RGB, max_values = data["models_RGB"], data["errors_RGB"], data["max_values"]
    else:
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
        x_flat, y_flat = x_grid.ravel(), y_grid.ravel()

        models_RGB = []
        errors_RGB = []
        max_values = []

        for c in range(3):
            print(f"拟合通道 {c + 1} 中...")
            V = img[:, :, c].ravel()
            model, poly = polynomial_fit(x_flat, y_flat, V, S)
            V_fit = evaluate_model(model, poly, x_flat, y_flat)
            mse = np.mean((V - V_fit) ** 2)

            idx_max = np.argmax(V_fit)
            x_max, y_max, V_max = x_flat[idx_max], y_flat[idx_max], V_fit[idx_max]
            max_values.append([x_max, y_max, V_max])

            models_RGB.append((model, poly))
            errors_RGB.append(mse)

            print(
                f"通道 {c + 1}: 最大值 V_max = {V_max:.4f}, (x_max, y_max) = ({x_max:.2f}, {y_max:.2f}), MSE: {mse:.2e}")

        save_to_json(param_file, models_RGB, errors_RGB, max_values)
        print(f"优化完成，参数已保存至 {param_file}。")

    # **绘制 3D 曲面图**
    print("开始绘制 3D 结果...")
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 5))
    colors = ['r', 'g', 'b']
    titles = ['Red Channel', 'Green Channel', 'Blue Channel']

    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    x_flat, y_flat = x_grid.ravel(), y_grid.ravel()

    for c in range(3):
        ax = axes[c]
        model, poly = models_RGB[c]
        V_fit = evaluate_model(model, poly, x_flat, y_flat)
        V_grid_fit = V_fit.reshape(H, W)

        sample_idx_hdr = np.linspace(0, len(x_flat) - 1, 500, dtype=int)
        ax.scatter(x_flat[sample_idx_hdr], y_flat[sample_idx_hdr], img[:, :, c].ravel()[sample_idx_hdr], c=colors[c],
                   s=3)
        ax.plot_surface(x_grid, y_grid, V_grid_fit, color=colors[c], alpha=0.6)
        ax.scatter(*max_values[c], color='k', marker='p', s=100)

        ax.set_title(f"{titles[c]} (MSE: {errors_RGB[c]:.2e})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        ax.view_init(30, 45)
        ax.grid(True)

    plt.show()
    print("绘制完成！")


if __name__ == "__main__":
    main()
