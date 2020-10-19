import cv2
import os
import numpy as np
import time
import threading
import multiprocessing

def get_images():
    images = []
    path = 'photos'
    myList = os.listdir(path)

    for img in myList:
        curImg = cv2.imread(f'{path}/{img}')
        curImg = cv2.resize(curImg, (640, 480))
        images.append(curImg)
    return images


def keypoints(img, res, sift):
    kp, des = sift.detectAndCompute(img, None)
    res.append(kp)
    res.append(des)

def find_kp_and_matrix(images):
    (image2, image1) = images
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # sift1 = cv2.SIFT_create()
    # sift2 = cv2.SIFT_create()
    # time1 = time.time()
    # res1, res2 = [], []
    # t1 = multiprocessing.Process(keypoints(gray1, res1, sift1))
    # t2 = multiprocessing.Process(keypoints(gray2, res2, sift2))
    # time2 = time.time()
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    # kp1, des1 = res1
    # kp2, des2 = res2
    # time3 = time.time()
    # print(kp1, des1)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    # time4 = time.time()
    # print('create_multi', time2-time1)
    # print('kp_multi', time3-time2)
    # print('kp_normal', time4-time3)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < n.distance:
            good.append(m)

    MIM_MATCH_COUNT = 10
    if len(good) <= MIM_MATCH_COUNT:
        print("not enough keypoints found")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M


def match_pano(images, M):
    (image2, image1) = images
    h, w, z = image1.shape
    dst = cv2.warpPerspective(image1, M, (image2.shape[1] + image1.shape[1], image2.shape[0]))
    s = 0
    while (s < w) and not np.sum(dst[0:h, s]):
        s += 1
    dst[:h, :s] = image2[:h, :s]
    for n in range(s, w):
        x = (n-s)/(w-s)
        up = 0
        while not np.sum(dst[up, n]):
            up += 5
        down = h-1
        while not np.sum(dst[down, n]):
            down -= 5
        dst[:up,n],dst[up:down,n],dst[down:,n] = image2[:up,n], cv2.addWeighted(image2[up:down, n], 1 - x, dst[up:down, n], x, 0), image2[down:, n]
    return dst


def stitch(images):
    matrix = find_kp_and_matrix(images)
    pano = match_pano(images, matrix)
    return pano


if __name__ == '__main__':
    images = get_images()
    t1 = time.time()
    matrix = find_kp_and_matrix(images)
    t2 = time.time()
    pano = match_pano(images, matrix)
    t3 = time.time()
    print("kp and matrix", t2-t1)
    print("matching", t3-t2)
    cv2.imshow('pano', pano)
    cv2.waitKey(0)
