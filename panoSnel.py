import cv2
import os
import numpy as np
import time
import multiprocessing
import threading

def fotos_ophalen():
    images = []
    path = 'photos'
    myList = os.listdir(path)
    for img in myList:
        curImg = cv2.imread(f'{path}/{img}')
        curImg = cv2.resize(curImg, (640, 480))
        images.append(curImg)
        # cv2.imshow(f'image{len(images)}', curImg)
        # cv2.waitKey(500)
    return images, (640, 480)

def methode1(images):

    stitcher = cv2.Stitcher.create()
    start = time.time()
    (status, result) = stitcher.stitch(images)
    end = time.time()
    print('methode 1:', end-start)
    if status == cv2.STITCHER_OK:
        print('panorama generated')
        cv2.imshow('result', result)
        # print(result.shape)
        # pan = result[40:400, 50:790]
        # cv2.imshow('panorama', pan)
    else:
        print('unsuccessfull')

# methode: 2

def keypoints_en_transformatiematrix(left, right):
    start = time.time()
    gray1 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)    # met zwart-wit werken is sneller
    gray2 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)      # zoek de keypoints
    kp2, des2 = sift.detectAndCompute(gray2, None)
    # cv2.imshow('original images keypoints', cv2.drawKeypoints(right, kp1, None))
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < n.distance:
            good.append(m)
    print(good)
    # draw_parameters = dict(matchColor=(0, 255, 0),
    #                        singlePointColor=None,
    #                        flags=2)
    # img3 = cv2.drawMatches(left, kp1, right, kp2, good, None, **draw_parameters)
    # cv2.imshow('draw_matches', img3)
    end = time.time()
    print("time for keypoints:", end-start)
    MIM_MATCH_COUNT = 10
    if len(good) <= MIM_MATCH_COUNT:
        print("not enough keypoints found")

    start = time.time()
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    end = time.time()
    print('transformatiematrix berekenen:', end-start)
    return M

def warpperspective(left, right, M):
    h, w, z = left.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    end = time.time()
    # dst = cv2.perspectiveTransform(pts, M)
    # img2 = cv2.polylines(left, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # cv2.imshow("original_image_overlapping.jpg", img2)
    dst = cv2.warpPerspective(right, M, (left.shape[1] + right.shape[1], right.shape[0]))
    return dst


def voeg_samen(dst, left, w, h, s):
    for n in range(s, w):
        x = (n-s)/(w-s)
        cv2.waitKey(1)
        up = 0
        while not np.sum(dst[up, n]):
            up += 5
        down = h-1
        while not np.sum(dst[down, n]):
            down -= 5
        dst[:up,n],dst[up:down,n],dst[down:,n] = left[:up, n], cv2.addWeighted(left[up:down, n], 1 - x, dst[up:down, n], x, 0), left[down:, n]


def voeg_samen_multiprocess(n):
    x = (n - s) / (w - s)
    up = 0
    while not np.sum(dst[up, n]):
        up += 5
    down = h - 1
    while not np.sum(dst[down, n]):
        down -= 5
    dst[:up, n], dst[up:down, n], dst[down:, n] = left_image[:up, n], cv2.addWeighted(left_image[up:down, n], 1 - x,dst[up:down, n], x, 0), left_image[down:, n]


def voeg_samen_multithread(left, dst, w, h, s):
    step = (w-s)//2
    t1 = threading.Thread(target=voeg_samen, args=(dst, left, s+step, h, s))
    t2 = threading.Thread(target=voeg_samen, args=(left, dst, w, h, s+step))

    t1.start()
    t2.start()

    t1.join()
    t2.join()





# maakt de overgang langzaam, waardoor er geen lijn zichtbaar is (snel maar niet zo goed)

# s = 0
# while (s < w) and not np.sum(dst[0:h, s]):
#     s += 1
# dst[:h, :s] = image2[:h, :s]
# for n in range(s, w):
#     x = (n-s)/(w-s)
#     new_col = cv2.addWeighted(image2[0:h, n], 1-x, dst[0:h, n], x, 0)
#     dst[0:h, n] = new_col

# merge = cv2.addWeighted(image2, 0.5, dst[0:image2.shape[0], 0:image2.shape[1]], 0.5, 0)
# dst[0:image2.shape[0], 0:image2.shape[1]] = image2

if __name__ == '__main__':

    (left_image, right_image), (w, h) = fotos_ophalen()
    # methode1(images)
    Matrix = keypoints_en_transformatiematrix(left_image, right_image)
    dst = warpperspective(left_image, right_image, Matrix)
    start = time.time()
    s = 0
    while (s < w) and not np.sum(dst[0:h, s]):
        s += 1
    dst[:h, :s] = left_image[:h, :s]
    # voeg_samen(dst, left_image, w, h, s)
    voeg_samen_multithread(left_image, dst, w, h, s)
    # pool = multiprocessing.Pool()
    # pool.map(voeg_samen_multiprocess, range(s, w))
    # processes = []
    # for n in range(s, w):
    #     p = multiprocessing.Process(target=voeg_samen_multiprocess, args=(n,s,w,h,left_image,dst))
    #     processes.append(p)
    #     p.start()
    # for process in processes:
    #     process.join()
    end = time.time()
    print("tijd om samen te voegen:", end-start)
    cv2.imshow("original_image_stiched_crop.jpg", dst)
    cv2.waitKey(0)
