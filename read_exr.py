import pyexr

# 读取 EXR 文件
exr_arq = pyexr.open(r"E:\sony_pictures\MTF_star_a7R3_100cm/merged_048_ARQ.exr")
exr_arw = pyexr.open(r"E:\sony_pictures\MTF_star_a7R3_100cm/merged_048_ARW.exr")

# # 获取所有通道
# channels_ = exr.channel_map.keys()
# print("通道列表:", channels)

# 读取所有通道数据
data_arq = exr_arq.get()
data_arw = exr_arw.get()
print("EXR 数据形状:", data.shape)

# 读取特定通道，例如 R、G、B
r_channel = exr.get("R")  # 读取 R 通道
g_channel = exr.get("G")  # 读取 G 通道
b_channel = exr.get("B")  # 读取 B 通道

print("R 通道形状:", r_channel.shape)
