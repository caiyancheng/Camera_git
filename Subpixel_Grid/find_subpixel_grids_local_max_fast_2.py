import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy import signal
from scipy.spatial import KDTree, cKDTree
import os
from tqdm import tqdm
# 1. 读取图像并预处理
def fast_peak_local_max(image, min_distance=3, threshold_abs=50):
    # 应用阈值过滤
    mask_above_thresh = image >= threshold_abs
    if not np.any(mask_above_thresh):
        return np.empty((0, 2), dtype=np.intp)
    # 创建结构元素（矩形对应棋盘距离）
    kernel_size = 2 * min_distance + 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 膨胀操作
    dilated = cv2.dilate(image, kernel)
    # 候选点：原图等于膨胀结果且超过阈值
    coords = np.column_stack(np.where((image == dilated) & mask_above_thresh))
    # coords = np.column_stack(np.where((image == dilated) & mask_above_thresh))
    if len(coords) == 0:
        return coords
    # 按强度降序排序
    intensities = image[coords[:, 0], coords[:, 1]]
    sorted_indices = np.argsort(-intensities)
    coords = coords[sorted_indices]

    mask = np.zeros(image.shape, dtype=bool)
    selected = []

    # 滑动窗口半径
    y_min = np.maximum(0, coords[:, 0] - min_distance)
    y_max = np.minimum(image.shape[0], coords[:, 0] + min_distance + 1)
    x_min = np.maximum(0, coords[:, 1] - min_distance)
    x_max = np.minimum(image.shape[1], coords[:, 1] + min_distance + 1)

    # 向量化处理
    for i in range(len(coords)):
        y, x = coords[i]
        if not mask[y, x]:
            selected.append([y, x])
            # 向量化屏蔽操作
            mask[y_min[i]:y_max[i], x_min[i]:x_max[i]] = True

    return np.array(selected)

input_path = r"E:\sony_pictures\Calibration_a7R3_on_display\100cm_4/DSC00000.png"
image = cv2.imread(input_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# 2. 检测局部极大值（调整参数以适应你的图像）
# coordinates = peak_local_max(gray, min_distance=3, threshold_abs=100)
coordinates = fast_peak_local_max(gray, min_distance=3, threshold_abs=170) #100 only checkerboard (F16), 220 only checkerboard (F8)
points = coordinates[:, [1, 0]].astype(float)  # (x,y)
if points.shape[0] != 42*65*16*16/2:
    raise ValueError("Wrong Number, Try Again!")

# 批量标记区域
mask = np.zeros_like(gray, dtype=np.uint8)
mask[coordinates[:, 0], coordinates[:, 1]] = 255
kernel = np.ones((1, 1), dtype=np.uint8)
expanded_mask = cv2.dilate(mask, kernel)
image[expanded_mask > 0] = [0, 0, 255]  # 标记为红色

# 自动生成输出路径（原路径加 _processed 后缀）
dir_name, file_name = os.path.split(input_path)
name, ext = os.path.splitext(file_name)
output_path = os.path.join(dir_name, f"{name}_processed_{ext}")  # 示例输出路径
cv2.imwrite(output_path, image)  # 保存处理结果
print(f"结果已保存至：{output_path}")