# 肤色检测
# 三种方法差别不大
# 不区分视频和图片
# 输入一张图片，返回分割后的图片
import cv2
import numpy as np

# 椭圆模型
def ellipse_detect(image):
    """
    :param image: 图片路径
    :return: None
    """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)

    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    cv2.imshow(image, img)
    dst = cv2.bitwise_and(img, img, mask=skin)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.imshow("cutout", dst)
    cv2.waitKey()

# YCrCb颜色空间Cr分量+Otsu法阈值分割
def cr_otsu(image):
    '''
    Y 表示明亮度，即灰阶值
    U V 表示的是色度
    Cr反映了RGB输入信号红色部分与RGB信号亮度值之间的差异。而Cb反映的是RGB输入信号蓝色部分与RGB信号亮度值之间的差异
    :param image:
    :return:
    '''
    img = image
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将RGB图像转换到YCrCb颜色空间，提取Cr分量图像
    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    # 对Cr做自二值化阈值分割处理（Otsu法）
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return skin

def crcb_range_sceening(image):
    """
    :param image: 图片路径
    :return: None
    """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)

    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    cv2.imshow(image, img)
    cv2.namedWindow(image + "skin2 cr+cb", cv2.WINDOW_NORMAL)
    cv2.imshow(image + "skin2 cr+cb", skin)

    dst = cv2.bitwise_and(img, img, mask=skin)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.imshow("cutout", dst)

    cv2.waitKey()

# image = "E:\\01.jpg"
# img = cv2.imread(image, cv2.IMREAD_COLOR)
# skin = cr_otsu(img)
# cv2.imshow('aa', skin)
# while True:
#     if cv2.waitKey(5) & 0xFF == 27:
#         break