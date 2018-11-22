import cv2 as cv
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# # random crop
#
# image = imread(r"C:\Users\pc\PycharmProjects\deeplearning-CNN-Week4- Neural Style Transfer\images\louvre.jpg")
# imshow(image)
# #plt.show()
#
# width = 800
# height = 600
# beta = 0.5
# delta = np.random.rand() * 2 - 1 # [-1, 1]
# width = int(width * np.sqrt(beta * (1 + delta)))
# height = int(height * np.sqrt(beta / (1 + delta)))
# point = [800 - width, 600 - height]
# print(point)
# start = [np.random.randint(0, point[0]), np.random.randint(0, point[1])]
# crop = image[start[0]:start[0] + width, start[1]:start[1] + height]
# imshow(crop)
# plt.show()


crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
'''
随机裁剪
area_ratio为裁剪画面占原画面的比例
hw_vari是扰动占原高宽比的比例范围
'''
def random_crop(img, area_ratio, hw_vari):
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta

    w_crop = int(round(w * np.sqrt(area_ratio * hw_mult)))
    if w_crop > w:
        w_crop = w
    h_crop = int(round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h
    # 随机生成左上角位置
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_image(img, x0, y0, w_crop, h_crop)

'''
定义旋转函数：
angle是逆时针旋转的角度
crop是个布尔值，表明是否要裁剪去除黑边
'''
def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    img_rotated = cv.warpAffine(img, M_rotate, (w, h))
    if crop:
        angle_crop = angle % 180  # 周期为180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0 #转化角度为弧度
        hw_ratio = float(h) / float(w) #计算高宽比
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1 # 计算分母项
        crop_mult = numerator / denominator

        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

'''
随机旋转
angle_vari是旋转角度的范围[-angle_vari, angle_vari)
p_crop是要进行去黑边裁剪的比例
'''

def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)

'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)

'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''

def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

'''
定义gamma变换函数：
gamma就是Gamma
'''

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)

'''
随机gamma变换
gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
'''

def random_gama_transform(img, gamma_vari):

    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


# image = imread(r"C:\Users\pc\PycharmProjects\deeplearning-CNN-Week4- Neural Style Transfer\images\louvre.jpg")
# plt.subplot(121)
# imshow(random_rotate(image, 60, 1))
# plt.subplot(122)
# imshow(random_rotate(image, 60, 0))
# plt.show()
# imshow(random_crop(image, 0.6, 1))
# plt.show()