import xmltodict
import cv2
import numpy as np

with open("car.xml") as f:
    xml = f.read()

result = xmltodict.parse(xml)
bndbox = result["annotation"]["object"]["bndbox"]

xmin = bndbox["xmin"]
ymin = bndbox["ymin"]
xmax = bndbox["xmax"]
ymax = bndbox["ymax"]

contours = np.array([[xmin, ymin], [xmin, ymax], [
                    xmax, ymax], [xmax, ymin]], np.int32)

img = cv2.imread("car.jpg")
cv2.fillConvexPoly(img, points=contours, color=(0, 0, 0))
cv2.imwrite("masked.jpg", img)
