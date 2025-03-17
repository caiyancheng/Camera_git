import os
import HDRutils
import numpy as np
import json
import matplotlib.pyplot as plt
import gzip
merge_exr = 0
plot_result = 1

image_root_path = r'E:\sony_pictures\Vignetting_2'
json_file_name = r'E:\sony_pictures\Vignetting_2_json/48_52_56_merge_focus_distance_100cm_real_distance_20cm.json.gz'

if merge_exr == 1:
    files = ['DSC00000_PSMS.ARQ', 'DSC00004_PSMS.ARQ']
    files = [os.path.join(image_root_path, i) for i in files]  # RAW input files
    HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True,
                             mtf_json='../MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
    Vignetting_map_R = HDR_img[:, :, 0] / HDR_img[:, :, 0].mean()
    Vignetting_map_G = HDR_img[:, :, 1] / HDR_img[:, :, 1].mean()
    Vignetting_map_B = HDR_img[:, :, 2] / HDR_img[:, :, 2].mean()
    json_data_dict = {
        'Vignetting_map_R': Vignetting_map_R.tolist(),
        'Vignetting_map_G': Vignetting_map_G.tolist(),
        'Vignetting_map_B': Vignetting_map_B.tolist()
    }
    with open(json_file_name, 'w') as outfile:
        json.dump(json_data_dict, outfile)
else:
    with gzip.open(json_file_name, 'rt', encoding='utf-8') as fp:
        json_data_dict = json.load(fp)
    Vignetting_map_R = np.array(json_data_dict['Vignetting_map_R'])
    Vignetting_map_G = np.array(json_data_dict['Vignetting_map_G'])
    Vignetting_map_B = np.array(json_data_dict['Vignetting_map_B'])

if plot_result == 1:
    downsample_factor = 10  # 例如 10 表示每 10 个像素取一个点
    Vignetting_map_R_ds = Vignetting_map_R[::downsample_factor, ::downsample_factor]
    Vignetting_map_G_ds = Vignetting_map_G[::downsample_factor, ::downsample_factor]
    Vignetting_map_B_ds = Vignetting_map_B[::downsample_factor, ::downsample_factor]

    # 归一化到 [0,1]，确保颜色亮度表示值的变化
    Vignetting_map_R_ds = (Vignetting_map_R_ds - Vignetting_map_R_ds.min()) / (Vignetting_map_R_ds.max() - Vignetting_map_R_ds.min())
    Vignetting_map_G_ds = (Vignetting_map_G_ds - Vignetting_map_G_ds.min()) / (Vignetting_map_G_ds.max() - Vignetting_map_G_ds.min())
    Vignetting_map_B_ds = (Vignetting_map_B_ds - Vignetting_map_B_ds.min()) / (Vignetting_map_B_ds.max() - Vignetting_map_B_ds.min())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制 R 通道
    axes[0].imshow(Vignetting_map_R_ds, cmap='Reds')
    axes[0].set_title("Vignetting Map - R Channel")
    axes[0].axis("off")

    # 绘制 G 通道
    axes[1].imshow(Vignetting_map_G_ds, cmap='Greens')
    axes[1].set_title("Vignetting Map - G Channel")
    axes[1].axis("off")

    # 绘制 B 通道
    axes[2].imshow(Vignetting_map_B_ds, cmap='Blues')
    axes[2].set_title("Vignetting Map - B Channel")
    axes[2].axis("off")

    output_path = "vignetting_maps_2D.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
