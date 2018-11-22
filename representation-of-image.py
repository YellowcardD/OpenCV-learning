import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
], dtype=np.uint8)
plt.imshow(img) ##matplotlib表示规则为RGB
plt.show()
##OpenCV表示规则为BGR
win = cv.namedWindow('image', flags=0) #flags=0 表示可以窗口可以用鼠标来改变大小
cv.imshow('image', img)
cv.waitKey(0) #第一个参数： 等待x ms，如果在此期间有按键按下，则立即结束并返回按下按键的ASCII码，否则返回-1 如果x=0，那么无限等待下去，直到有按键按下