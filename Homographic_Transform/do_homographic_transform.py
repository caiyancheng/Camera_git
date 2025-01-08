import os
import numpy as np
import cv2
import json

k = 0.0001 # 最小单位为多少m？

with open('homography_aruco_result.json', 'r') as fp:
    homography_aruco_result = json.load(fp)
obj_points = np.array(homography_aruco_result['obj_points'])
obj_points[:,0] += 1.211 / 2
obj_points[:,1] = -obj_points[:,1]
obj_points[:,1] += 0.681 / 2

obj_points = obj_points / k
img_points = np.array(homography_aruco_result['img_points'])
display_points = obj_points[:,:2]
homography_matrix, mask = cv2.findHomography(img_points, display_points)

display_width = round(1.211 / k)
display_height = round(0.681 / k)
display_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)

img_path = r'E:\sony_pictures\Homographic_1/DSC00002.png'
image = cv2.imread(img_path)
height, width, _ = image.shape
aligned_image = cv2.warpPerspective(image, homography_matrix, (display_width, display_height))

cordinate_center_x = round(1.211 / 2 / k)
# cordinate_center_y = aligned_image.shape[0] - round(0.7 / 2 / k)
cordinate_center_y = round(0.681 / 2 / k)
radius = 20
cv2.circle(aligned_image, (cordinate_center_x, cordinate_center_y ), radius, (0, 0, 255), -1)

cv2.imwrite('aligned_image.png', aligned_image)

