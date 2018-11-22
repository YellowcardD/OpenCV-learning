import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#存取图像
'''
#读取图像
color_img = cv.imread(r'pc\PycharmProjects\ex3\flower_photos\sunflowers\26254755_1bfc494ef1_n.jpg')
print(color_img.shape)

#读取单通道
gray_img = cv.imread(r'pc\PycharmProjects\ex3\flower_photos\sunflowers\26254755_1bfc494ef1_n.jpg', cv.IMREAD_GRAYSCALE)
print(gray_img.shape)

#把单通道图片保存后，再读取，仍然是3通道，相当于把单通道值复制到3个通道保存。
cv.imwrite('test_grayscale.jpg', gray_img)
reload_grayscale = cv.imread('test_grayscale.jpg')
print(reload_grayscale.shape)

#cv.IMWRITE_JPEG_QUALITY指定jpg质量，范围0-100， 默认95， 越高画质越好，文件越大。
cv.imwrite('test_imwrite.jpg', color_img, (cv.IMWRITE_JPEG_QUALITY, 80))
#cv.IMWRITE_PNG_COMPRESSION指定png质量，范围0-9， 默认3， 越高文件越小，画质越差
cv.imwrite('test_imwrite.png', color_img, (cv.IMWRITE_PNG_COMPRESSION, 5))
'''

#缩放，裁剪和补边
'''
img = cv.imread(r'\PycharmProjects\ex3\flower_photos\sunflowers\26254755_1bfc494ef1_n.jpg') #读取240x320的照片
img_120x120 = cv.resize(img, (120, 120)) #缩放成120x120的方形图像。

#不直接指定缩放后的大小，通过fx和fy指定缩放比例，0.5则长宽都为原来的一半
#等效于img_120*240 = cv2.resize(img, (240, 120))，注意指定大小的格式是（宽度，高度）
#插值方法默认是cv2.INTER_LINEAR, 这里指定为最近邻插值。
img_120x160 = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)

#在上张图片的基础上，上下各贴20像素的黑边，生成160x160的图像
img_160x160 = cv.copyMakeBorder(img_120x160, 20, 20, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))

#对照片中的部分进行剪裁
patch = img[20:150, -180:-50]

cv.imwrite('cropped.jpg', patch)
cv.imwrite('resize_120x120.jpg', img_120x120)
cv.imwrite('resize_120x160.jpg', img_120x160)
cv.imwrite('resize_160x160.jpg', img_160x160)
'''

#色调 明暗 直方图和Gamma曲线
'''
img = cv.imread(r'E:\graduate\Computer Vision\flower_photos\sunflowers\26254755_1bfc494ef1_n.jpg')
#通过cvtColor把图像从BGR转换到HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # hsv为一种颜色空间(h:hue为色调，s:saturation为饱和度， v:value为明度)
#h中，绿色比黄色的值高一点，所以给每个像素+15， 黄色的树叶会变绿
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0] + 15) % 180 #h的取值范围为[0, 180)
turn_green_img = cv.cvtColor(turn_green_hsv, cv.COLOR_HSV2BGR)
cv.imwrite('turn_green.jpg', turn_green_img)
#减小饱和度会让图像损失鲜艳，变得更灰
colorless_hsv = img_hsv.copy()
colorless_hsv[:, :, 1] = 0.5 * colorless_hsv[:, :, 1] #s的取值范围为[0, 256)
colorless_img = cv.cvtColor(colorless_hsv, cv.COLOR_HSV2BGR)
cv.imwrite('colorless.jpg', colorless_img)
#减小明度为原来的一半
darker_hsv = img_hsv.copy()
darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2] #v的取值范围为[0, 256)
darker_img = cv.cvtColor(darker_hsv, cv.COLOR_HSV2BGR)
cv.imwrite('darker.jpg', darker_img)

#分通道计算每个通道的直方图
hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv.calcHist([img], [2], None, [256], [0, 256])

#定义gamma矫正的函数
def gamma_trans(img, gamma):

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    #实现这个映射用的是OpenCV的查表函数
    return cv.LUT(img, gamma_table)

#执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升。
img_corrected = gamma_trans(img, 0.5)
cv.imwrite('gamma_corrented.jpg', img_corrected)

#分通道计算Gamma矫正后的直方图
hist_b_corrected = cv.calcHist([img_corrected], [0], None, [256], [0, 256])
hist_g_corrected = cv.calcHist([img_corrected], [1], None, [256], [0, 256])
hist_r_corrected = cv.calcHist([img_corrected], [2], None, [256], [0, 256])

fig = plt.figure()
pix_hists = [
    [hist_b, hist_g, hist_r],
    [hist_b_corrected, hist_g_corrected, hist_r_corrected]
]
pix_vals = range(256)
for sub_plt, pix_hist in zip([121, 122], pix_hists):
    ax = fig.add_subplot(sub_plt, projection='3d')
    for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
        cs = [c] * 256
        ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='None', lw=0)

    ax.set_xlabel('Pixel Values')
    ax.set_xlim([0, 256])
    ax.set_ylabel('Channels')
    ax.set_zlabel('Counts')

plt.show()
'''
