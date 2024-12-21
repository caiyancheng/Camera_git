import cv2

pattern_id = []

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_image = cv2.aruco.generateImageMarker(dictionary, pattern_id, 200)
cv2.imwrite(f"marker{pattern_id}.png", marker_image)
