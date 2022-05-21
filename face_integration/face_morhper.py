import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay


predictor_model = '../shape_predictor_68_face_landmarks.dat'


def get_points(image):  # 用 dlib 来得到人脸的特征点

    face_detector = dlib.get_frontal_face_detector()  # 正向人脸检测器，进行人脸检测，提取人脸外部矩形框
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)  # 获取landmark
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])

    # 加入四个顶点和四条边的中点
    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)


def get_triangles(points):  #  在特征点上使用 Delaunay 三角剖分，将点集连接成一定大小的三角形，且分配要相对合理，才能呈现出漂亮的三角化
    return Delaunay(points).simplices


def affine_transform(input_image, input_triangle, output_triangle, size):  # 对人脸进行仿射变换，确定位置
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image


def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):  # 三角形变形，Alpha 混合
    # 计算三角形的边界框
    rect1 = cv2.boundingRect(np.float32([tri1]))  # 寻找tri1的左上角坐标，和tri1的长和宽
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    # 在边界框内进行仿射变换
    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                     rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    # 加权求和
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    # 生成模板
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    # 应用模板
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask


def morph_faces(filename1, filename2, alpha=0.5):  # 融合图片
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_CUBIC)
    print('img1.shape',img1.shape)
    print('img2.shape',img2.shape)

    points1 = get_points(img1)
    print('pionts1:',len(points1),points1)
    points2 = get_points(img2)
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    import pandas as pd
    p = pd.DataFrame(points)
    p.to_csv('./1.csv')

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    triangles = get_triangles(points)
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)

    return np.uint8(img_morphed)


def main(file1,file2,alpha):
    try:
     alpha = float(alpha)
    except:
        alpha = 0.5
    img_morphed = morph_faces(file1, file2, alpha)
    output_file = '{}_{}_{}.jpg'.format(
        file1.split('.')[0][-2:], file2.split('.')[0][-1:], alpha)
    cv2.imwrite(output_file, img_morphed)
    return output_file


