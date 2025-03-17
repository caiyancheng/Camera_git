import imageio
import imageio.v3 as io, rawpy, pyexr
import cv2
png_file_name = r'E:\sony_pictures\Color_fringing_1/DSC00000.png'
img_data_io = io.imread(png_file_name)
img_data_cv2 = cv2.imread(png_file_name)
X = 1
