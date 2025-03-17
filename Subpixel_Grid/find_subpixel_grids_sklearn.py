import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 读取图像并转换为灰度
image = cv2.imread('subpixel_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理（根据实际情况调整阈值）
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 查找轮廓并计算质心
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))
centroids = np.array(centroids)

# 定义自适应计算EPS的函数
def calculate_eps(data, percentile=50):
    sorted_data = np.sort(data, axis=0)
    diffs = np.diff(sorted_data, axis=0)
    non_zero_diffs = diffs[diffs > 0]
    return np.percentile(non_zero_diffs, percentile) * 2 if len(non_zero_diffs) > 0 else 1

# 对y坐标进行聚类（行检测）
y_data = centroids[:, 1].reshape(-1, 1)
eps_y = calculate_eps(y_data)
clustering_y = DBSCAN(eps=eps_y, min_samples=1).fit(y_data)
rows_y = [np.mean(y_data[clustering_y.labels_ == label]) for label in np.unique(clustering_y.labels_)]
rows_y.sort()

# 对x坐标进行聚类（列检测）
x_data = centroids[:, 0].reshape(-1, 1)
eps_x = calculate_eps(x_data)
clustering_x = DBSCAN(eps=eps_x, min_samples=1).fit(x_data)
cols_x = [np.mean(x_data[clustering_x.labels_ == label]) for label in np.unique(clustering_x.labels_)]
cols_x.sort()

# 计算网格线位置
vertical_lines = [(rows_y[i] + rows_y[i+1])/2 for i in range(len(rows_y)-1)]
horizontal_lines = [(cols_x[j] + cols_x[j+1])/2 for j in range(len(cols_x)-1)]

# 绘制网格线
output_image = image.copy()
for y in vertical_lines:
    cv2.line(output_image, (0, int(y)), (output_image.shape[1], int(y)), (0, 255, 0), 1)
for x in horizontal_lines:
    cv2.line(output_image, (int(x), 0), (int(x), output_image.shape[0]), (255, 0, 0), 1)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), plt.title('With Grid Lines')
plt.show()