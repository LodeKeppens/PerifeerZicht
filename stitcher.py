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
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M


def stitch_frame(images, M, s=0):
    (image2, image1) = images
    h, w, d = image1.shape
    dst = cv2.warpPerspective(image1, M, (2*w, h))
    dst[:h, :s] = image2[:h, :s]
    step = 1
    for n in range(s, w-step, step):
        x = (n - s) / (w - s)
        dst[:, n] = cv2.addWeighted(image2[:, n:n+step], 1 - x, dst[:, n:n+step], x, 0)
    return dst


def eerste_frame(images):
    left, right = images
    h, w, d = left.shape
    matrix = find_kp_and_matrix((left, right))
    if matrix is None:
        return None, None
    dst = cv2.warpPerspective(right, matrix, (2*w, h))
    s = 0
    while (s < w) and not np.sum(dst[:, s]):
        s += 1
    return matrix, s


def stitch_video(left_video, right_video):
    keypoints_found = False
    i = -1
    while not keypoints_found:
        i += 1
        M, s = eerste_frame((left_video[i], right_video[i]))
        if M is not None:
            keypoints_found = True
    pano = [stitch_frame((left_video[i], right_video[i]), M, s)]
    for n in range(0, len(left_video)):
        new_frame = stitch_frame((left_video[n], right_video[n]), M, s)
        pano.append(new_frame)
    return pano
