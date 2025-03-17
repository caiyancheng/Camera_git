import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义线性空间的RGB值（如157，253，315）
rgb_values = np.array([271, 343, 286])

# 归一化RGB值，确保它们在[0, 255]之间
rgb_values_normalized = np.clip(rgb_values, 0, 255) / 400

# Gamma校正，通常使用2.2的标准值
gamma = 2.2
rgb_values_gamma_corrected = np.power(rgb_values_normalized, 1 / gamma)

# 创建一个新的图形
fig, ax = plt.subplots()

# 生成一个矩形并填充颜色
rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=1, edgecolor='black', facecolor=rgb_values_gamma_corrected)
ax.add_patch(rect)
ax.text(0.5, 0.5, 'Measurement', ha='center', va='center', fontsize=20, color='black')
# 设置坐标轴的显示范围，确保矩形完整显示
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

# 隐藏坐标轴
ax.axis('off')

# 显示图形
# plt.tight_layout()
# plt.title("Rectangle Filled with Gamma Corrected RGB Color")
plt.tight_layout()
plt.show()