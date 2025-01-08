import rawpy

# 加载 RAW 文件
# file_path = r"E:\sony_pictures\MTF_star_multi_exposure_80cm/DSC00002.ARW"
file_path = r"E:\sony_pictures\Calibration\120cm/DSC00000.ARW" # 替换为实际路径
with rawpy.imread(file_path) as raw:
    # 提取黑电平信息
    black_level_per_channel = raw.black_level_per_channel  # 每通道的黑电平

    # 获取拜尔阵列模式
    bayer_pattern = raw.raw_colors_visible  # 拜尔阵列模式 (0: 红, 1: 绿, 2: 蓝)

    # 提取拜尔模式描述符
    bayer_desc = raw.color_desc.decode('utf-8')  # e.g., "RGBG" 或 "RGGB"

    # HDRutils中提取数据
    img_1 = raw.raw_image_visible
    img_2 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, user_wb=None,
                          user_flip=0, output_color=rawpy.ColorSpace.sRGB)
    X = 1
