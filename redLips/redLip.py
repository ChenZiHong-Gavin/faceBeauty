# 红唇
import mediapipe as mp
import cv2
import numpy as np

# 填充嘴唇
def change_color_lip(img, list_lms, index_lip_up, index_lip_down, color):
    mask = np.zeros_like(img)

    points_lip_up = list_lms[index_lip_up, :]
    mask = cv2.fillPoly(mask, [points_lip_up], (255, 255, 255))

    points_lip_down = list_lms[index_lip_down, :]
    mask = cv2.fillPoly(mask, [points_lip_down], (255, 255, 255))

    img_color_lip = np.zeros_like(img)
    img_color_lip[:] = color
    # cv2.imshow("color lip",img_color_lip)
    # cv2.imshow("mask",mask)

    img_color_lip = cv2.bitwise_and(mask, img_color_lip)
    # cv2.imshow("color lip",img_color_lip)
    # 高斯模糊
    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)
    # alpha融合
    img_color_lip = cv2.addWeighted(img, 1, img_color_lip, 0.3, 0)
    return img_color_lip


def empty(a):
    pass

def get_lip_picture(image):
    # 创建人脸关键点检测对象
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    img = image
    # 口红颜色调节 b g r
    # 口紫
    # color = (255, 0, 0)
    # color = (255, 255, 0)
    # 小辣椒色
    color = (41, 48, 178)
    # 正红色
    # color = (13, 3, 194)
    # 牛血红
    # color = (0, 5, 106)
    # 小羊皮暖柿红
    # color = (71, 93, 239)
    # 获取宽度和高低
    image_height, image_width, _ = np.shape(img)
    # BGR 转 RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_RGB)
    list_lms = []
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        for i in range(478):
            pos_x = int(face_landmarks.landmark[i].x * image_width)
            pos_y = int(face_landmarks.landmark[i].y * image_height)
            list_lms.append((pos_x, pos_y))
        list_lms = np.array(list_lms, dtype=np.int32)
        # 嘴唇坐标
        index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80, 191,
                        78, 61]
        index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181,
                          91, 146, 61, 78]
        # img = change_color_lip(img,list_lms,index_lip_up,index_lip_down,color)

        img = change_color_lip(img, list_lms, index_lip_up, index_lip_down, color)
        return img
    # cv2.namedWindow("BGR", cv2.WINDOW_NORMAL)
    # cv2.imshow("BGR", img)
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     # 按键 "q" 退出
    #     if key == ord('q'):
    #         break

def get_lip_video(video_in, video_out):
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
                img_out = get_lip_picture(img)  # 逐帧处理
                writer.write(img_out)
            except:
                continue
            print("yes")
    else:
        print('视频打开失败！')
    writer.release()
    return 0
# filename = "05.jpg"
# image = cv2.imread("E:\\" + filename, cv2.IMREAD_COLOR)
# dst = get_lip_picture(image)
# cv2.imwrite("E:\\results\\redlip\\" + filename, dst)

filename = "E:\\06.MOV"
outFilename = "E:\\results\\redlip\\06.avi"

dst = get_lip_video(filename, outFilename)