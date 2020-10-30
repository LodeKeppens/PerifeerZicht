import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def video_ophalen(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    video = []
    while ret:
        # cv2.imshow('video', frame)
        frame = cv2.resize(frame, (640, 480))
        video.append(frame)
        ret, frame = cap.read()
    return video


def video_afspelen(frames):
    for frame in frames:
        cv2.imshow('video', frame)
        cv2.waitKey(33)


def splits(frames, w):
    left_video, right_video = [], []
    for frame in frames:
        left_frame, right_frame = frame[:, :2 * w // 3], frame[:, w // 3:]
        left_video.append(left_frame)
        right_video.append(right_frame)
    return left_video, right_video


def find_kp_and_matrix(images):
    (image2, image1) = images
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < n.distance:
            good.append(m)

    MIM_MATCH_COUNT = 10
    status = False
    if len(good) <= MIM_MATCH_COUNT:
        status = True

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, status


def match_pano(images, M):
    (image2, image1) = images
    h, w, a = image1.shape
    dst = cv2.warpPerspective(image1, M, (image2.shape[1] + image1.shape[1], image2.shape[0]))
    s = 0
    while (s < w) and not np.sum(dst[0:h, s]):
        s += 1
    dst[:h, :s] = image2[:h, :s]
    step = 50
    for n in range(s, w-step, step):
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
        dst[:, n:n+step] = cv2.addWeighted(image2[:, n:n+step], 1 - x, dst[:, n:n+step], x, 0)
    return dst


def stitch_image(images):
    matrix = find_kp_and_matrix(images)
    pano = match_pano(images, matrix)
    return pano


def stitch_video(left_video, right_video):
    t1 = time.time()
    pano = []
    matrix, status = find_kp_and_matrix((left_video[0], right_video[0]))
    t2 = time.time()
    print('tijd voor transformatiematrix:', t2-t1)
    new_frame = match_pano((left_video[0], right_video[0]), matrix)
    tijden1 = [t2-t1]
    for n in range(0, 100):
        t1 = time.time()
        new_frame = match_pano((left_video[n], right_video[n]), matrix)
        pano.append(new_frame[:, :640])
        tijden1.append(time.time()-t1)
    # print(tijden1)
    return pano, tijden1


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


video = video_ophalen('video/test.3gp')
left, right = splits(video, 640)
print("met zelf gemaakt programma:\n")
t1 = time.time()
pano, tijden1 = stitch_video(left, right)
t2 = time.time()
x = range(len(tijden1))
# plt.subplot(221)
# plt.semilogy(x, tijden1, label="cv2.stitch")
plt.scatter(x,tijden1, label="snelle_methode")
plt.xlabel("frame")
plt.ylabel("tijd")
plt.ylim(0, 0.1)
plt.xlim(0, len(tijden1))
plt.show()
# input("press enter to watch video")
t4 = time.time()
# video_afspelen(pano)
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
