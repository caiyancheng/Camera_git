import HDRutils

image_path = r'E:\sony_pictures\MTF_star/DSC00000.ARW'

image_data = HDRutils.imread(image_path)
HDRutils.imwrite(image_path.replace('.ARW', '.exr'), image_data)