import os

import mediapipe as mp
import csv
VISUALIZE_FACE_POINTS = False
filters_config = {
    'anonymous':
        [{'path': "AR_filter/filters/anonymous.png",
          'anno_path': "AR_filter/filters/anonymous_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anime':
        [{'path': "AR_filter/filters/anime.png",
          'anno_path': "AR_filter/filters/anime_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "AR_filter/filters/dog-ears.png",
          'anno_path': "AR_filter/filters/dog-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "AR_filter/filters/dog-nose.png",
          'anno_path': "AR_filter/filters/dog-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "AR_filter/filters/cat-ears.png",
          'anno_path': "AR_filter/filters/cat-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "AR_filter/filters/cat-nose.png",
          'anno_path': "AR_filter/filters/cat-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "AR_filter/filters/jason-joker.png",
          'anno_path': "AR_filter/filters/jason-joker_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'gold-crown':
        [{'path': "AR_filter/filters/gold-crown.png",
          'anno_path': "AR_filter/filters/gold-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "AR_filter/filters/flower-crown.png",
          'anno_path': "AR_filter/filters/flower-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'green-carnival':
        [{'path': "AR_filter/filters/green-carnival.png",
          'anno_path': "AR_filter/filters/green-carnival_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}


# detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]
    multi_filter_runtime = []
    for filter in filters:
        temp_dict = {}
        img1, img1_alpha = load_filter_img( filter['path'], filter['has_alpha'])
        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha
        points = load_landmarks(filter['anno_path'])
        temp_dict['points'] = points
        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue
        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap
        multi_filter_runtime.append(temp_dict)
    return filters, multi_filter_runtime


def get_AR_Picture(image, name="dog"):
    # 变量设置
    count = 0
    isFirstFrame = True
    sigma = 50
    filters, multi_filter_runtime = load_filter(name)
    frame = image
    points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 如果脸被挡住了，或者脸不全
    if not points2 or (len(points2) != 75):
        print("脸被挡住了")
        return image

    img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points2Prev = np.array(points2, np.float32)
    img2GrayPrev = np.copy(img2Gray)

    lk_params = dict(winSize=(101, 101), maxLevel=15,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
    points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                    np.array(points2, np.float32),
                                                    **lk_params)

    # Final landmark points are a weighted average of detected landmarks and tracked landmarks

    for k in range(0, len(points2)):
        d = cv2.norm(np.array(points2[k]) - points2Next[k])
        alpha = math.exp(-d * d / sigma)
        points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
        points2[k] = constrainPoint(points2[k], frame.shape[1], frame.shape[0])
        points2[k] = (int(points2[k][0]), int(points2[k][1]))


    if VISUALIZE_FACE_POINTS:
        for idx, point in enumerate(points2):
            cv2.circle(frame, point, 2, (255, 0, 0), -1)
            cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        cv2.imshow("landmarks", frame)

    for idx, filter in enumerate(filters):
        filter_runtime = multi_filter_runtime[idx]
        img1 = filter_runtime['img']
        points1 = filter_runtime['points']
        img1_alpha = filter_runtime['img_a']
        if filter['morph']:
            hullIndex = filter_runtime['hullIndex']
            dt = filter_runtime['dt']
            hull1 = filter_runtime['hull']

            # create copy of frame
            warped_img = np.copy(frame)

            # Find convex hull
            hull2 = []
            for i in range(0, len(hullIndex)):
                hull2.append(points2[hullIndex[i][0]])

            mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
            mask1 = cv2.merge((mask1, mask1, mask1))
            img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

            # Warp the triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                warpTriangle(img1, warped_img, t1, t2)
                warpTriangle(img1_alpha_mask, mask1, t1, t2)

            # Blur the mask before blending
            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

            mask2 = (255.0, 255.0, 255.0) - mask1

            # Perform alpha blending of the two images
            temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
            output = temp1 + temp2
        else:
            dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
            tform = similarityTransform(list(points1.values()), dst_points)
            # Apply similarity transform to input image
            trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
            trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
            mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

            # Blur the mask before blending
            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

            mask2 = (255.0, 255.0, 255.0) - mask1

            # Perform alpha blending of the two images
            temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
            output = temp1 + temp2

        frame = output = np.uint8(output)

    return output

def test(name):
    print(os.getcwd()+"\\AR_filter\\")

def get_AR_video(video_in, video_out):
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
                img_out = get_AR_Picture(img,'cat')  # 逐帧处理
                writer.write(img_out)
            except:
                continue
            print("yes")
    else:
        print('视频打开失败！')
    writer.release()
    return 0

import cv2
import numpy as np
import math


# Constrains points to be inside boundary
def constrainPoint(p, w, h):
  p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  return p

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int(xin), np.int(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int(xout), np.int(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]


# Check if a point is inside a rectangle
def rectContains(rect, point):
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[2]:
    return False
  elif point[1] > rect[3]:
    return False
  return True


# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(rect, points):

  # Create an instance of Subdiv2D
  subdiv = cv2.Subdiv2D(rect)

  # Insert points into subdiv
  for p in points:
    subdiv.insert((int(p[0]), int(p[1])))

  # Get Delaunay triangulation
  triangleList = subdiv.getTriangleList()

  # Find the indices of triangles in the points array
  delaunayTri = []

  for t in triangleList:
    # The triangle returned by getTriangleList is
    # a list of 6 coordinates of the 3 points in
    # x1, y1, x2, y2, x3, y3 format.
    # Store triangle as a list of three points
    pt = []
    pt.append((t[0], t[1]))
    pt.append((t[2], t[3]))
    pt.append((t[4], t[5]))

    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
      # Variable to store a triangle as indices from list of points
      ind = []
      # Find the index of each vertex in the points list
      for j in range(0, 3):
        for k in range(0, len(points)):
          if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
            ind.append(k)
        # Store triangulation as a list of indices
      if len(ind) == 3:
        delaunayTri.append((ind[0], ind[1], ind[2]))

  return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

  # Given a pair of triangles, find the affine transform.
  warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

  # Apply the Affine Transform just found to the src image
  dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

  return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
  # Find bounding rectangle for each triangle
  r1 = cv2.boundingRect(np.float32([t1]))
  r2 = cv2.boundingRect(np.float32([t2]))

  # Offset points by left top corner of the respective rectangles
  t1Rect = []
  t2Rect = []
  t2RectInt = []

  for i in range(0, 3):
    t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
    t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

  # Get mask by filling triangle
  mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

  # Apply warpImage to small rectangular patches
  img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

  size = (r2[2], r2[3])

  img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

  img2Rect = img2Rect * mask

  # Copy triangular region of the rectangular patch to the output image
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

# filename = "05.jpg"
# src = cv2.imread('E:\\'+filename)
# magic_face = get_AR_Picture(src, "flower-crown")
# cv2.imwrite("E:\\results\\AReffect\\" + filename, magic_face)

# cv2.namedWindow("BGR", cv2.WINDOW_NORMAL)
# cv2.imshow("BGR", magic_face)
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     # 按键 "q" 退出
#     if key == ord('q'):
#         break

# filename = "E:\\06.MOV"
# outFilename = "E:\\results\\AReffect\\06.avi"
#
# dst = get_AR_video(filename, outFilename)