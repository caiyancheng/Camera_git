import os
import HDRutils
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
merge_exr = 1
plot_result = 1

image_root_path = r'E:\sony_pictures\Vignetting_2'
save_root_path = r'E:\sony_pictures\Vignetting_2_json'
os.makedirs(save_root_path, exist_ok=True)
json_file_name_list = ['0_4_8_merge_focus_distance_100cm_real_distance_100cm.json',
                       '12_16_20_merge_focus_distance_100cm_real_distance_60cm.json',
                       '24_28_32_merge_focus_distance_100cm_real_distance_40cm.json',
                       '36_40_44_merge_focus_distance_100cm_real_distance_30cm.json',
                       '48_52_56_merge_focus_distance_100cm_real_distance_20cm.json']
files_list = [['DSC00000_PSMS.ARQ', 'DSC00004_PSMS.ARQ', 'DSC00008_PSMS.ARQ'],
              ['DSC00012_PSMS.ARQ', 'DSC00016_PSMS.ARQ', 'DSC00020_PSMS.ARQ'],
              ['DSC00024_PSMS.ARQ', 'DSC00028_PSMS.ARQ', 'DSC00032_PSMS.ARQ'],
              ['DSC00036_PSMS.ARQ', 'DSC00040_PSMS.ARQ', 'DSC00044_PSMS.ARQ'],
              ['DSC00048_PSMS.ARQ', 'DSC00052_PSMS.ARQ', 'DSC00056_PSMS.ARQ'],]
out_figure_name_list = ['0_4_8_merge_focus_distance_100cm_real_distance_100cm.png',
                        '12_16_20_merge_focus_distance_100cm_real_distance_60cm.png',
                        '24_28_32_merge_focus_distance_100cm_real_distance_40cm.png',
                        '36_40_44_merge_focus_distance_100cm_real_distance_30cm.png',
                        '48_52_56_merge_focus_distance_100cm_real_distance_20cm.png']

for index in tqdm(range(len(files_list))):
    files = files_list[index]
    json_file_name = os.path.join(save_root_path, json_file_name_list[index])
    out_figure_name = os.path.join(save_root_path, out_figure_name_list[index])
    if merge_exr == 1:
        files = [os.path.join(image_root_path, i) for i in files]  # RAW input files
        HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True,
                                 mtf_json='../MTF/mtf_sony_a7R3_FE_28_90_100cm.json')[0]
        Vignetting_map_R = HDR_img[:, :, 0] / HDR_img[:, :, 0].mean()
        Vignetting_map_G = HDR_img[:, :, 1] / HDR_img[:, :, 1].mean()
        Vignetting_map_B = HDR_img[:, :, 2] / HDR_img[:, :, 2].mean()
        json_data_dict = {}
        json_data_dict['Vignetting_map_R'] = Vignetting_map_R.tolist()
        json_data_dict['Vignetting_map_G'] = Vignetting_map_G.tolist()
        json_data_dict['Vignetting_map_B'] = Vignetting_map_B.tolist()
        with open(json_file_name, 'w') as outfile:
            json.dump(json_data_dict, outfile)
    else:
        with open(json_file_name, 'r') as fp:
            json_data_dict = json.load(fp)
        Vignetting_map_R = np.array(json_data_dict['Vignetting_map_R'])
        Vignetting_map_G = np.array(json_data_dict['Vignetting_map_G'])
        Vignetting_map_B = np.array(json_data_dict['Vignetting_map_B'])

    if plot_result == 1:
        downsample_factor = 10  # 例如10表示每10个像素取一个点
        H, W = Vignetting_map_R.shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        X_ds = X[::downsample_factor, ::downsample_factor]
        Y_ds = Y[::downsample_factor, ::downsample_factor]
        Z_R_ds = Vignetting_map_R[::downsample_factor, ::downsample_factor]
        Z_G_ds = Vignetting_map_G[::downsample_factor, ::downsample_factor]
        Z_B_ds = Vignetting_map_B[::downsample_factor, ::downsample_factor]
        fig = plt.figure(figsize=(18, 6))

        # 绘制 R 通道
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X_ds, Y_ds, Z_R_ds, cmap='Reds', edgecolor='none')
        ax1.set_title('Vignetting Map - R Channel')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        ax1.set_zlabel('Value / Mean')
        ax1.set_zlim(0, 2)

        # 绘制 G 通道
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X_ds, Y_ds, Z_G_ds, cmap='Greens', edgecolor='none')
        ax2.set_title('Vignetting Map - G Channel')
        ax2.set_xlabel('Width')
        ax2.set_ylabel('Height')
        ax2.set_zlabel('Value / Mean')
        ax2.set_zlim(0, 2)

        # 绘制 B 通道
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X_ds, Y_ds, Z_B_ds, cmap='Blues', edgecolor='none')  # 绿色，避免 B 通道过深
        ax3.set_title('Vignetting Map - B Channel')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
        ax3.set_zlabel('Value / Mean')
        ax3.set_zlim(0, 2)

        plt.savefig(out_figure_name, dpi=300, bbox_inches='tight')
        # plt.tight_layout()
        # plt.show()


