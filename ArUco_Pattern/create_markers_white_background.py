import cv2
import numpy as np

# 创建白色背景图像
background_color = 255  # 白色背景 (255代表白色)
image_size = 400  # 设置背景图像的尺寸

# 创建一个全白的背景
background = np.ones((image_size, image_size), dtype=np.uint8) * background_color

# 设置Aruco标记的字典和ID
pattern_id = 0  # 你可以修改为你想要的ID
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 生成Aruco标记图像
marker_image = cv2.aruco.generateImageMarker(dictionary, pattern_id, 200)

# 计算标记的放置位置（使其居中）
x_offset = (background.shape[1] - marker_image.shape[1]) // 2
y_offset = (background.shape[0] - marker_image.shape[0]) // 2

# 将Aruco标记放置到白色背景上
background[y_offset:y_offset + marker_image.shape[0], x_offset:x_offset + marker_image.shape[1]] = marker_image

# 保存最终图像
cv2.imwrite(f"Markers/marker{pattern_id}_on_white.png", background)

# 显示生成的图像
# cv2.imshow("Aruco Marker", background)

# # 获取屏幕的宽度和高度
# screen_width = 38400  # 示例分辨率（请根据您的显示器调整）
# screen_height = 2160  # 示例分辨率（请根据您的显示器调整）
#
# # 计算窗口的左上角坐标，使其居中
# window_width = background.shape[1]
# window_height = background.shape[0]
# x_center = (screen_width - window_width) // 2
# y_center = (screen_height - window_height) // 2
#
# # 设置显示窗口位置
# cv2.moveWindow("Aruco Marker", x_center, y_center)
#
# # 等待按键，按任意键关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()
