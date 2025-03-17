import cv2
import numpy as np
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy import signal
from scipy.spatial import KDTree, cKDTree, ConvexHull
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 膨胀操作
    dilated = cv2.dilate(image, kernel)
    # 候选点：原图等于膨胀结果且超过阈值
    coords = np.column_stack(np.where((image == dilated) & mask_above_thresh))
    if len(coords) == 0:
        return coords
    # 按强度降序排序
    intensities = image[coords[:, 0], coords[:, 1]]
    sorted_indices = np.argsort(-intensities)
    coords = coords[sorted_indices]

    mask = np.zeros(image.shape, dtype=bool)
    selected = []

    # 滑动窗口半径
    r = min_distance
    for y, x in coords:
        if not mask[y, x]:
            selected.append([y, x])
            # 计算屏蔽区域边界
            y0 = max(0, y - r)
            y1 = min(image.shape[0], y + r + 1)
            x0 = max(0, x - r)
            x1 = min(image.shape[1], x + r + 1)
            # 向量化屏蔽操作
            mask[y0:y1, x0:x1] = True

    return np.array(selected)


def register_checkerboard(points):
    """
    输入：
      points: (N,2) 的 numpy 数组，每行是一个白色子像素在摄像图像中的 (x, y) 坐标。
             总数应为 25×20×8×8/2 = 16000 个点。

    输出：
      registration: (N,4) 的 numpy 数组，每行为一个点对应的四个整数：
          [global_row, global_col, subpixel_row, subpixel_col]
      其中 global_row/global_col 表示该白块在棋盘中（仅白块）的全局行列编号，
      subpixel_row/subpixel_col 表示该子像素在对应白块中的行列位置（取值 0~7）。
    """
    # ----- 第一步：将 16000 个白子像素聚成 250 个白块簇  -----
    n_blocks = 250
    kmeans_blocks = KMeans(n_clusters=n_blocks, random_state=0).fit(points)
    block_labels = kmeans_blocks.labels_  # 长度为 N，每个点的簇号（0～249）
    block_centers = kmeans_blocks.cluster_centers_  # 每个簇的中心 (250,2)

    # ----- 第二步：确定各白块在整棋盘中的全局序号  -----
    # 我们知道棋盘总共有 20 行（棋盘 20 行块），
    # 但由于棋盘交错，白块分布在每一行中个数可能不同（偶数行13个，奇数行12个）。
    # 为了获得“行号”，先将 250 个簇中心在棋盘平面中做 PCA，
    # 利用第二主成分作为“垂直方向”，然后对该方向做 1D 聚类得到 20 个行。
    mean_center = np.mean(block_centers, axis=0)
    centers_centered = block_centers - mean_center
    # SVD 得到主方向（假设第一主分量沿棋盘较长方向，即水平方向）
    U, S, Vt = np.linalg.svd(centers_centered)
    proj = centers_centered.dot(Vt.T)  # 将每个中心转换到新坐标系下, shape=(250,2)
    # proj[:,1] 近似代表“竖直方向”（棋盘行方向）

    # 利用 KMeans 将 250 个中心按 proj[:,1] 分成 20 类，每一类对应一行
    n_rows = 20
    kmeans_rows = KMeans(n_clusters=n_rows, random_state=0).fit(proj[:, 1].reshape(-1, 1))
    row_labels = kmeans_rows.labels_  # 每个白块所属行的标签（0～19，但顺序不一定从上到下）

    # 为保证行号从上到下，我们计算每个聚类在 proj[:,1] 上的均值，并排序
    row_means = np.array([np.mean(proj[row_labels == i, 1]) for i in range(n_rows)])
    sorted_row_ids = np.argsort(row_means)  # 排序后，第 0 个为最上面的一行
    # 建立从原来的 row_labels 到全局行号的映射
    row_label_to_global = {old_label: global_row for global_row, old_label in enumerate(sorted_row_ids)}

    # 对于同一行内的白块，再按照 proj[:,0]（大致水平方向）排序，得到从左到右的顺序
    # 保存每个白块对应的全局（行号, 列号）——列号为该行内的序号
    block_global_indices = {}
    for i in range(n_blocks):
        r_lab = row_labels[i]
        global_row = row_label_to_global[r_lab]
        # 找出与 i 同行的所有白块（它们的索引）
        indices_in_row = np.where(row_labels == r_lab)[0]
        # 按 proj[:,0] 排序（大致水平坐标），得到从左到右的顺序
        sorted_in_row = indices_in_row[np.argsort(proj[indices_in_row, 0])]
        # 在这一行中，i 的“列号”即为其在 sorted_in_row 中的位置
        global_col = int(np.where(sorted_in_row == i)[0][0])
        block_global_indices[i] = (global_row, global_col)

    # ----- 第三步：在每个白块簇内对 64 个子像素进行局部配准  -----
    N = points.shape[0]
    registration = np.empty((N, 4), dtype=int)  # 保存最终结果

    # 对每个簇分别处理
    for block_idx in range(n_blocks):
        # 找出属于该簇的所有点（应为 64 个）
        pt_indices = np.where(block_labels == block_idx)[0]
        pts_block = points[pt_indices]  # (64,2)

        # 利用“角点”方法获得该簇四角对应的候选点
        # 这里利用简单的数值组合： x+y 最小的当作左上角，x+y最大的为右下角，
        # x-y 最小的为右上角，x-y最大的为左下角。
        s = pts_block[:, 0] + pts_block[:, 1]
        top_left = pts_block[np.argmin(s)]
        bottom_right = pts_block[np.argmax(s)]
        d = pts_block[:, 0] - pts_block[:, 1]
        top_right = pts_block[np.argmin(d)]
        bottom_left = pts_block[np.argmax(d)]

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        # 理想中，白块内部的 8×8 点映射后应落在坐标系 [0,7]×[0,7] 的格点上
        dst = np.array([[0, 0],
                        [7, 0],
                        [7, 7],
                        [0, 7]], dtype=np.float32)
        # 计算透视变换矩阵（注意：对于理想的平面棋盘，变换可用 cv2.getPerspectiveTransform 获得）
        H = cv2.getPerspectiveTransform(src, dst)
        # 对该簇内所有点做透视变换，得到“理想”局部坐标
        pts_trans = cv2.perspectiveTransform(pts_block.reshape(-1, 1, 2), H).reshape(-1, 2)
        # 四舍五入取整得到子像素的行列编号（取值 0～7）
        subpixel_cols = np.rint(pts_trans[:, 0]).astype(int)
        subpixel_rows = np.rint(pts_trans[:, 1]).astype(int)
        # 防止边界问题，限定在 0～7 内
        subpixel_cols = np.clip(subpixel_cols, 0, 7)
        subpixel_rows = np.clip(subpixel_rows, 0, 7)

        # 取得该白块在棋盘中的全局编号
        global_row, global_col = block_global_indices[block_idx]

        # 将结果写回到 registration 数组中
        for idx, pt_idx in enumerate(pt_indices):
            registration[pt_idx, 0] = global_row
            registration[pt_idx, 1] = global_col
            registration[pt_idx, 2] = subpixel_rows[idx]
            registration[pt_idx, 3] = subpixel_cols[idx]

    return registration

input_path = r"E:\sony_pictures\Calibration_a7R3_on_display\50cm/DSC00001.png"
image = cv2.imread(input_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# 2. 检测局部极大值（调整参数以适应你的图像）
# coordinates = peak_local_max(gray, min_distance=3, threshold_abs=100)
coordinates = fast_peak_local_max(gray, min_distance=3, threshold_abs=220) #100 only checkerboard (F16), 220 only checkerboard (F8)
points = coordinates[:, [1, 0]].astype(float)  # (x,y)
if points.shape[0] != 20*25*8*8/2:
    raise ValueError("Wrong Number, Try Again!")

reg = register_checkerboard(points)
X = 1
print(reg[:10])

# 批量标记区域
mask = np.zeros_like(gray, dtype=np.uint8)
mask[coordinates[:, 0], coordinates[:, 1]] = 255
kernel = np.ones((3, 3), dtype=np.uint8)
expanded_mask = cv2.dilate(mask, kernel)
image[expanded_mask > 0] = [0, 0, 255]  # 标记为红色

# 自动生成输出路径（原路径加 _processed 后缀）
dir_name, file_name = os.path.split(input_path)
name, ext = os.path.splitext(file_name)
output_path = os.path.join(dir_name, f"{name}_processed_{ext}")  # 示例输出路径
cv2.imwrite(output_path, image)  # 保存处理结果
print(f"结果已保存至：{output_path}")