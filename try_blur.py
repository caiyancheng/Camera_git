import cv2

# 读取图片
image = cv2.imread(r'E:\sony_pictures\ArUco_Homo_1/DSC00002.png')
blurred_image = cv2.GaussianBlur(image, (17, 17), 0)
cv2.imwrite(r'E:\sony_pictures\ArUco_Homo_1/blurred_image.jpg', blurred_image)
