# 对皮肤进行美白
# 使用颜色序列
from PIL import Image, ImageEnhance
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import getSkin
import faceDect.faceDetect as faceDect
import numpy as np

Color_list = [
    1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
    41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
    76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
    106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
    130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
    151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
    171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
    217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
    228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
    238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
    245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
    251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
    254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 256]


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
                img_out = face_white_picture(img)  # 逐帧处理
                writer.write(img_out)
            except:
                continue
            print("yes")
    else:
        print('视频打开失败！')
    writer.release()
    return 0


# 图片方法
def face_white_picture(image):
    img = image
    height, width, n = image.shape
    newFace = image.copy()
    for i in range(height):
        for j in range(width):
            b = newFace[i, j, 0]
            g = newFace[i, j, 1]
            r = newFace[i, j, 2]
            newFace[i, j, 0] = Color_list[b]
            newFace[i, j, 1] = Color_list[g]
            newFace[i, j, 2] = Color_list[r]
    img2 = Image.fromarray(newFace)
    # 锐度调节
    enh_img = ImageEnhance.Sharpness(img2)
    # 锐化操作的factor是一个0-2的浮点数，当factor=0时，返回一个完全模糊的图片对象，当factor=1时，返回一个完全锐化的图片对象
    image_sharped = enh_img.enhance(1.6)
    # 颜色均衡调节
    con_img = ImageEnhance.Contrast(image_sharped)
    image_con = con_img.enhance(1.2)
    white_face = np.asarray(image_con)
    return white_face

# filename = "05.jpg"
# image = cv2.imread("E:\\" + filename, cv2.IMREAD_COLOR)
# dst = face_white_picture(image)
# cv2.imwrite("E:\\results\\whitening\\" + filename, dst)

# filename = "E:\\06.MOV"
# outFilename = "E:\\results\\whitening\\06.avi"
#
# dst = face_smooth_video(filename, outFilename)
