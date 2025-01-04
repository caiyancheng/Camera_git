import numpy as np
import cv2

k = 1000
image_width = round(29.7 * k)
image_height = round(21 * k)

pure_image = np.ones((image_height, image_width), dtype=np.uint8) * 200

cv2.imwrite('pure_image.png', pure_image)
