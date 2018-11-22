import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#图像的仿射变换 即对图像缩放，旋转，剪切，翻转和平移进行组合。

img = cv.imread(r'E:\graduate\Computer Vision\flower_photos\sunflowers\26254755_1bfc494ef1_n.jpg')

#沿着横纵轴放大1.6倍， 然后平移(-30, -40), 最后沿原图大小截取，等效于裁剪并放大
# M_crop = np.array([
#     [1.6, 0, -30],
#     [0, 1.6, -40]
# ], dtype=np.float32)
# img_crop = cv.warpAffine(img, M_crop, (320, 240))
# cv.imwrite('img_crop.jpg', img_crop)

# x轴的剪切变换，角度15°
theta = 15 * np.pi / 180
print(theta)
# M_shear = np.array([
#     [1, np.tan(theta),0],
#     [0, 1, 0]
# ], dtype=np.float32)
# img_sheared = cv.warpAffine(img, M_shear, (320, 240))
# cv.imwrite('img_sheared.jpg', img_sheared)

# 顺时针旋转，角度15°
M_rotate = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0]
], dtype=np.float32)
img_rotated = cv.warpAffine(img, M_rotate, (320, 240))
cv.imwrite('img_rotated.jpg', img_rotated)

# 某种变换，具体旋转+缩放+旋转组合可以通过SVD分解理解
# M = np.array([
#     [1, 1.5, -100],
#     [0.5, 2, -30]
# ],dtype=np.float32)
# img_transformed = cv.warpAffine(img, M, (320, 240))
# cv.imwrite('img_transfered.jpg', img_transformed)
