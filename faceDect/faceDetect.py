import cv2
import mediapipe as mp
import time


def handleVideo(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    pTime = 0

    with mp_face_mesh.FaceMesh(
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            # Draw the face mesh annotations on the image.
            # and print landmarks' id, x, y, z
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 在图片上画mask
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    # print id, x, y, z
                    # time cost
                    # for id,lm in enumerate(face_landmarks.landmark):
                    #     ih, iw, ic = image.shape
                    #     x,y = int(lm.x*iw), int(lm.y*ih)
                    #     print(id, x,y,lm.z)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # cv2.putText(image, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image

# 获得脸的轮廓
def getFaceRectangle(image):
    # 输出检测到的人脸集合，包括一个边界框和6个关键点
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                return results.detections[0].location_data.relative_bounding_box

# image = cv2.imread("E:\\02.jpg", cv2.IMREAD_COLOR)
# # 人脸区域检测
# dst = getFaceRectangle(image)
#
# cv2.namedWindow("1", cv2.WINDOW_NORMAL)
# cv2.imshow('1', dst)
# while True:
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
