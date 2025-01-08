from PIL import Image

def generate_color_image(color, width, height, output_filename):
    # 创建指定颜色的图片
    img = Image.new("RGB", (width, height), color)
    img.save(output_filename)
    return output_filename, width, height

# 定义颜色和分辨率
color = (32, 32, 32)  # RGB 值 (1, 1, 1)
width, height = 3840, 2160  # 替换为第二屏幕的分辨率
output_filename = "white_32.png"

# 生成图片
filename, w, h = generate_color_image(color, width, height, output_filename)
print(f"生成的图片已保存为 {filename}, 分辨率为 {w}x{h}")
