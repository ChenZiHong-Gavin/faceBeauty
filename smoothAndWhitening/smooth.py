# 对皮肤进行磨皮
# 图片方法提供双边滤波
# 视频方法直接使用高斯模糊
from PIL import Image, ImageEnhance
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

import faceDect.faceDetect as faceDect
import numpy as np



# 视频方法
def face_smooth_video(video_in, video_out):
    '''
        双边滤波方法
        src：原图像；
        d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；d越小保留的细节越多
        sigmaColor：颜色空间的标准方差，一般尽可能大；
        sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。
    '''
    # 视频
    video_dir = video_in
    video_out_dir = video_out
    cap = cv2.VideoCapture(video_dir)  # 生成读取视频对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_out_dir, fourcc, 30.0, (1440, 1080), True)
    if cap.isOpened():
        while True:
            ret, img = cap.read()  # img 就是一帧图片
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if not ret: break  # 当获取完最后一帧就结束
            try:
                img_out = face_smooth_picture(img)  # 逐帧处理
                writer.write(img_out)
            except:
                continue
            print("yes")
    else:
        print('视频打开失败！')
    writer.release()
    return 0


# 图片方法
def face_smooth_picture(image):
    img = image
    # 获取脸的轮廓
    imgWidth, imgHeight, channel = image.shape
    rectangle = faceDect.getFaceRectangle(img)
    # 起点
    rect_start_point = _normalized_to_pixel_coordinates(
        rectangle.xmin, rectangle.ymin, imgHeight,
        imgWidth)
    rect_end_point = _normalized_to_pixel_coordinates(
        rectangle.xmin + rectangle.width,
        rectangle.ymin + rectangle.height, imgHeight,
        imgWidth)
    start1 = 0
    start2 = 0
    if rect_start_point[1] - (rect_end_point[1] - rect_start_point[1]) * 0.25 < 0:
        start1 = 0
    else:
        start1 = rect_start_point[1] - int((rect_end_point[1] - rect_start_point[1]) * 0.25)
    if rect_start_point[0] - (rect_end_point[0] - rect_start_point[0]) * 0.25 < 0:
        start2 = 0
    else:
        start2 = rect_start_point[0] - int((rect_end_point[0] - rect_start_point[0]) * 0.25)
    # 设置感兴趣区域
    face = img[start1: rect_end_point[1], start2:rect_end_point[0]]
    # 只对脸部处理
    smooth_face = cv2.bilateralFilter(face, 0, 50, 10)
    img.flags.writeable = True
    img[start1: rect_end_point[1], start2:rect_end_point[0]] = smooth_face
    height, width, n = img.shape
    img2 = img.copy()
    # for i in range(height):
    #     for j in range(width):
    #         b = img2[i, j, 0]
    #         g = img2[i, j, 1]
    #         r = img2[i, j, 2]
    #         img2[i, j, 0] = Color_list[b]
    #         img2[i, j, 1] = Color_list[g]
    #         img2[i, j, 2] = Color_list[r]
    img2 = Image.fromarray(img2)
    # 锐度调节
    enh_img = ImageEnhance.Sharpness(img2)
    # 锐化操作的factor是一个0-2的浮点数，当factor=0时，返回一个完全模糊的图片对象，当factor=1时，返回一个完全锐化的图片对象
    image_sharped = enh_img.enhance(1.6)
    # 颜色均衡调节
    con_img = ImageEnhance.Contrast(image_sharped)

    image_con = con_img.enhance(1.2)
    img = np.asarray(image_con)
    return img


# filename = "04.jpg"
# image = cv2.imread("E:\\" + filename, cv2.IMREAD_COLOR)
# dst = face_smooth_picture(image)
# cv2.imwrite("E:\\results\\smooth\\" + filename, dst)

# filename = "E:\\06.MOV"
# outFilename = "E:\\results\\smooth\\06.avi"
#
# dst = face_smooth_video(filename, outFilename)
