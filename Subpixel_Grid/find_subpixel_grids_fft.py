import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = r"E:\sony_pictures\Calibration_a7R3_on_display\50cm/DSC00001.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #[5320,7968]

# 计算 FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# 计算 FFT 频谱的峰值位置（估算像素间距）
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# 寻找主频率峰值点
peak_threshold = np.max(magnitude_spectrum) * 0.5  # 设定阈值
indices = np.where(magnitude_spectrum > peak_threshold)
distances = np.sqrt((indices[0] - crow) ** 2 + (indices[1] - ccol) ** 2)
sorted_indices = np.argsort(distances)
primary_frequencies = np.array(indices).T[sorted_indices[:4]]

# 计算主要像素间距
pixel_spacing_x = np.abs(primary_frequencies[1][1] - ccol)
pixel_spacing_y = np.abs(primary_frequencies[1][0] - crow)

# 生成网格线
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap="gray")

for i in range(0, cols, int(pixel_spacing_x)):
    ax.axvline(i, color='r', linestyle='--', linewidth=0.5)

for j in range(0, rows, int(pixel_spacing_y)):
    ax.axhline(j, color='r', linestyle='--', linewidth=0.5)

plt.title("Subpixel Grid Overlay")
plt.show()
