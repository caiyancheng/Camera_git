import cv2
import rawpy
import numpy as np
import os
# 读取 ARQ 文件
root_path = r'E:\sony_pictures\a7R3_100_aruco_4_demosaic'
arq_file = r'DSC00000_PSMS.ARQ' #四个通道就是RGBG
arq_data = rawpy.imread(os.path.join(root_path, arq_file))
raw_image_visible = arq_data.raw_image_visible.copy()  # (5320, 7968, 4), uint16

# 提取四个通道
channels = np.array([raw_image_visible[:, :, i] for i in range(4)])
channels_normalized = (channels / channels.max() * 255).astype(np.uint8)

# 逐个通道保存
# for i, channel in enumerate(channels_normalized):
#     cv2.imwrite(os.path.join(root_path, f'DSC00000_PSMS_channel_{i+1}.png'), channel)

# 组合成三通道图像
three_channel_image = np.stack([channels_normalized[0],
                                ((channels_normalized[1]+channels_normalized[3])/2).astype(np.uint8),
                                channels_normalized[2]], axis=-1)
three_channel_image = cv2.cvtColor(three_channel_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(root_path, 'DSC00000_PSMS.png'), three_channel_image)