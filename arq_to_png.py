import rawpy
import imageio

# 读取 ARW 文件
arw_file_list = ["DSC00000.ARW", "DSC00001.ARW", "DSC00002.ARW", "DSC00003.ARW"]
if len(arw_file_list) != 4:
    raise ValueError("The length of arw_file_list should be 4.")

for arw_file in arw_file_list:
    

# 使用 rawpy 进行解码
with rawpy.imread(arq_file) as raw:
    # 进行去马赛克（debayer/demosaic）
    rgb_image = raw.postprocess()

# 保存为 PNG
output_file = "output.png"
imageio.imwrite(output_file, rgb_image)

print(f"ARQ 文件已成功转换为 PNG: {output_file}")
