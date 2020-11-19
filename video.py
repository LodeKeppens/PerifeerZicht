import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def video_ophalen(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    video = []
    while ret:
        frame = cv2.resize(frame, (640, 480))
        video.append(frame)
        ret, frame = cap.read()
    return video


def video_afspelen(frames, name='video'):
    for frame in frames:
        cv2.imshow(name, frame)
        cv2.waitKey(40)


def splits(frames, w):
    left_video, right_video = [], []
    for frame in frames:
        left_frame, right_frame = frame[:, :2 * w // 3], frame[:, w // 3:]
        left_video.append(left_frame)
        right_video.append(right_frame)
    return left_video, right_video


def find_kp_and_matrix(images):
    (left, right) = images
    # gray1 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(right, None)
    # cv2.imshow('keypoints left image:', cv2.drawKeypoints(left, kp2, None))
    # cv2.imshow('keypoints right image:', cv2.drawKeypoints(right, kp1, None))
    # cv2.waitKey(0)
    print('aangal keypoints:', len(des1), len(des2))
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print('aantal matches:', len(good))
    MIM_MATCH_COUNT = 10
    if len(good) <= MIM_MATCH_COUNT:
        return None
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M


def stitch_frame(images, M, s=0):
    t1 = time.time()
    (left, right) = images
    h, w, a = right.shape
    dst = cv2.warpPerspective(right, M, (left.shape[1] + right.shape[1], left.shape[0]))
    t2 = time.time()
    dst[:h, :s] = left[:h, :s]
    step = 1
    for n in range(s, w - 1):
        x = (n - s) / (w - s)
        # up = 0
        # while up < h // 2 and not np.sum(dst[up, n]):
        #     up += 1
        # down = h - 1
        # while down > h // 2 and not np.sum(dst[down, n]):
        #     down -= 1
        # dst[:up, n], dst[up:down, n], dst[down:, n] = image2[:up, n], \
        #                                               cv2.addWeighted(image2[up:down, n], 1 - x, dst[up:down, n], x, 0), \
        #                                               image2[down:, n]
        dst[:, n] = cv2.addWeighted(left[:, n], 1 - x, dst[:, n], x, 0)
    t3 = time.time()
    return dst, t2 - t1, t3 - t2


def stitch_frame(images, M, s=0, pos_left_bottom_corner = (0,0)):
    t1 = time.time()
    (left, right) = images
    h, w, a = right.shape
    dst = cv2.warpPerspective(right, M, (left.shape[1] + right.shape[1], left.shape[0]))
    t2 = time.time()
    k = 0
    dst[:h, :s + k] = left[:h, :s + k]


    cv2.circle(dst,pos_left_bottom_corner,5,(0,0,255),thickness=2)

    #
    # b = find_first_nonzero_pixel_in_col(dst, pos_left_bottom_corner[1] + 1)
    # a = pos_left_bottom_corner[0]
    # slope = b - a
    # print('slope:', slope)
    # print('a:',a)
    # print('b:',b)
    # print('h,w,a:', h,w,a)
    # print()
    # for row in range(a, h):
    #     for col in range(pos_left_bottom_corner[1],pos_left_bottom_corner[1]+row//slope):
    #         dst[row,col] = left[row,col]


    step = 1
    # for n in range(s, w-1):
    #     x = (n - s) / (w - s)
    # up = 0
    # while up < h // 2 and not np.sum(dst[up, n]):
    #     up += 1
    # down = h - 1
    # while down > h // 2 and not np.sum(dst[down, n]):
    #     down -= 1
    # dst[:up, n], dst[up:down, n], dst[down:, n] = image2[:up, n], \
    #                                               cv2.addWeighted(image2[up:down, n], 1 - x, dst[up:down, n], x, 0), \
    #                                               image2[down:, n]
    # dst[:, n] = cv2.addWeighted(left[:, n], 1 - x, dst[:, n], x, 0)
    t3 = time.time()
    return dst, t2 - t1, t3 - t2


def eerste_frame(images):
    left, right = images
    t1 = time.time()
    matrix = find_kp_and_matrix((left, right))
    t2 = time.time()
    if matrix is None:
        return None, None
    print('tijd voor transformatiematrix', t2 - t1)
    dst = cv2.warpPerspective(right, matrix, (left.shape[1] + right.shape[1], left.shape[0]))
    leftmost_col = 0
    while (leftmost_col < 640) and not np.sum(dst[:, leftmost_col]):
        leftmost_col += 1

    pos_left_bottom_corner = find_pos_left_bottom_corner(dst, leftmost_col)
    return matrix, leftmost_col, pos_left_bottom_corner


def find_pos_left_bottom_corner(matrix, leftmost_col):
    pos_left_bottom_corner = (0, 0)
    cur_col = leftmost_col
    cur_intersection = find_first_nonzero_pixel_in_col(matrix, cur_col)
    next_intersection = find_first_nonzero_pixel_in_col(matrix, cur_col + 1)
    slope = next_intersection - cur_intersection
    while (cur_col <= leftmost_col + 30) and pos_left_bottom_corner == (0, 0):
        current_slope = next_intersection - cur_intersection
        if current_slope != slope :
            pos_left_bottom_corner = (next_intersection, cur_col + 1)
            print(pos_left_bottom_corner)
        else:
            cur_col += 1
            cur_intersection = next_intersection
            next_intersection = find_first_nonzero_pixel_in_col(matrix, cur_col + 1)
        cv2.circle(matrix,(cur_col+1,next_intersection),4,(0,0,255))
        cv2.imshow('Find leftbottom pixel',matrix)
    return pos_left_bottom_corner


def find_first_nonzero_pixel_in_col(matrix, col):
    nb_rows, _, _ = matrix.shape
    row = nb_rows
    nonzero = False
    while row >= 0 and not nonzero:
        row -= 1
        nonzero = sum(matrix[row, col])

    return row

def stitch_video(left_video, right_video):
    t1 = time.time()
    keypoints_found = False
    i = -1
    while not keypoints_found:
        i += 1
        M, s, pos_left_bottom_corner = eerste_frame((left_video[i], right_video[i]))
        if M is not None:
            keypoints_found = True

    pano = []
    t2 = time.time()
    tijden1 = [t2 - t1]
    tijden2, tijden3 = [], []
    for n in range(0, len(left_video)):
        t1 = time.time()
        new_frame, tTrans, tSamen = stitch_frame((left_video[n], right_video[n]), M, s, pos_left_bottom_corner)
        cv2.imshow('video', new_frame)
        pano.append(new_frame)
        tijden1.append(time.time() - t1)
        tijden2.append(tTrans)
        tijden3.append(tSamen)
    # print(tijden1)
    return pano, tijden1, tijden2, tijden3


def stitch_cv2(left_video, right_video):
    stitcher = cv2.Stitcher.create()
    t1 = time.time()
    tijden2 = []
    for n in range(0, 100):
        t1 = time.time()
        new_frame = stitcher.stitch((left_video[n], right_video[n]))
        pano.append(new_frame)
        tijden2.append(time.time() - t1)
    print(tijden2)
    return pano, tijden2


# video = video_ophalen('video/test.3gp')
# left, right = splits(video, 640)
right = video_ophalen("video's/vid_R.avi")
left = video_ophalen("video's/vid_L.avi")
# video_afspelen(left, 'left')
# video_afspelen(right, 'right')
print("met zelf gemaakt programma:\n")
t1 = time.time()
pano, tijden1, tijden2, tijden3 = stitch_video(left, right)
t2 = time.time()
x = range(len(tijden1))
# plt.subplot(221)
# plt.semilogy(x, tijden1, label="cv2.stitch")
plt.scatter(x, tijden1, label="totaal")
x = range(len(tijden2))
plt.scatter(x, tijden2, label="transformatie")
plt.scatter(x, tijden3, label="samenvoegen")
plt.legend()
plt.xlabel("frame")
plt.ylabel("tijd")
plt.ylim(0, 0.02)
plt.xlim(0, len(tijden1))
# plt.show()
# input("press enter to watch video")
t4 = time.time()
video_afspelen(pano)
t3 = time.time()
# print('this took', t2 - t1, 'seconds, for a video of', t3 - t4, 'seconds')
cv2.destroyAllWindows()
# print("met stitcher functie:\n")
# # input("press enter")
# pano, tijden2 = stitch_cv2(left, right)
# # video_afspelen(pano)
# x = range(len(tijden2))
# # punten2 = plt.scatter(x,tijden2, marker="*", color="green", label="cv2.stitch")
# # plt.subplot(221)
# plt.semilogy(x, tijden2, label="vaste_punten")
# plt.xlabel("frame")
# plt.ylabel("tijd [s]")
# plt.ylim(0.001, 10)
# plt.xlim(0,len(tijden2))
# plt.show()
