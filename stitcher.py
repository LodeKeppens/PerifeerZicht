import cv2
import numpy as np


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


def stitch_frame(images, M, s=0):
    (image2, image1) = images
    h, w, d = image1.shape
    dst = cv2.warpPerspective(image1, M, (2*w, h))
    dst[:h, :s] = image2[:h, :s]
    step = 1
    for n in range(s, w-1):
        x = (n - s) / (w - s)
        dst[:, n] = cv2.addWeighted(image2[:, n], 1 - x, dst[:, n], x, 0)
    return dst


def eerste_frame(images):
    left, right = images
    matrix, status = find_kp_and_matrix((left, right))
    dst = cv2.warpPerspective(right, matrix, (left.shape[1] + right.shape[1], left.shape[0]))
    s = 0
    while (s < 640) and not np.sum(dst[:, s]):
        s += 1
    return matrix, s


def stitch_video(left_video, right_video):
    M, s = eerste_frame((left_video[0], right_video[0]))
    pano = [stitch_frame((left_video[0], right_video[0]), M, s)[0]]
    for n in range(0, len(left_video)):
        new_frame = stitch_frame((left_video[n], right_video[n]), M, s)
        pano.append(new_frame)
    return pano
