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
        if m.distance < 0.7*n.distance:
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
    for n in range(s, w-1):
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
    return dst, t2-t1, t3-t2


def eerste_frame(images):
    left, right = images
    t1 = time.time()
    matrix = find_kp_and_matrix((left, right))
    t2 = time.time()
    if matrix is None:
        return None, None
    print('tijd voor transformatiematrix', t2-t1)
    dst = cv2.warpPerspective(right, matrix, (left.shape[1] + right.shape[1], left.shape[0]))
    s = 0
    while (s < 640) and not np.sum(dst[:, s]):
        s += 1
    return matrix, s


def stitch_video(left_video, right_video):
    t1 = time.time()
    keypoints_found = False
    i = -1
    while not keypoints_found:
        i += 1
        M, s = eerste_frame((left_video[i], right_video[i]))
        if M is not None:
            keypoints_found = True

    pano = []
    t2 = time.time()
    tijden1 = [t2-t1]
    tijden2, tijden3 = [], []
    for n in range(0, len(left_video)):
        t1 = time.time()
        new_frame, tTrans, tSamen = stitch_frame((left_video[n], right_video[n]), M, s)
        cv2.imshow('video',new_frame)
        pano.append(new_frame)
        tijden1.append(time.time()-t1)
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
        new_frame = stitcher.stitch((left_video[n],right_video[n]))
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
plt.scatter(x,tijden1, label="totaal")
x = range(len(tijden2))
plt.scatter(x,tijden2,label="transformatie")
plt.scatter(x,tijden3,label="samenvoegen")
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
