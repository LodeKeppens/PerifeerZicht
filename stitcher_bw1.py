import cv2
import numpy as np


def find_kp_and_matrix(images):
    """
    :param images: tuple met linker- en rechterfoto, vlolgorde is belangrijk!
    :return: berekent de transformatiematrix om de rechterfoto te transformeren
             zodat deze op de linkerfoto past om een panoramabeeld te voremen
    """

    # zet beelden om in zwart-wit om berekeningen sneller te maken
    (left, right) = images
    gray1 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    # bereken de keypoints in beide foto's
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # match de gevonden keypoints van de beide foto's
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # return de waarde None als er niet genoeg matches zijn gevonden
    MIM_MATCH_COUNT = 10
    if len(good) <= MIM_MATCH_COUNT:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # bereken de transformatiematrix op basis van de gevonden keypoints
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M


def stitch_frame(images, M, s=0):
    """
    :param images: tuple met linker- en rechterfoto, is belangrijk!
    :param M: transformatiematrix
    :param s: vanaf welke kolom de overlap start
     parameters M, s moeten worden berekend door de functie eerste frame
    :return: voegt de twee foto's uit images samen met de transformatiematrix,
             geeft als resultaat het panoramabeeld
    """

    # warpperspective van de rechterfoto
    (left, right) = images
    h, w, d = right.shape
    dst = cv2.warpPerspective(right, M, (2*w, h))

    # voeg de linkerfoto toe op het deel dat niet overlapt
    dst[:h, :s] = left[:h, :s]

    # voeg de foto's samen in het overlappend deel zodat een zachte overgang ontstaat
    step = 1
    for n in range(s, w-step, step):
        x = (n - s) / (w - s)
        dst[:, n] = cv2.addWeighted(left[:, n:n+step], 1 - x, dst[:, n:n+step], x, 0)
    return dst



def stitch_frame_right_warped(images, M, s=0):
    """
    :param images: tuple met linker- en rechterfoto, is belangrijk!
    :param M: transformatiematrix
    :param s: vanaf welke kolom de overlap start
     parameters M, s moeten worden berekend door de functie eerste frame
    :return: voegt de twee foto's uit images samen met de transformatiematrix,
             geeft als resultaat het panoramabeeld
    """

    # warpperspective van de rechterfoto
    # al gebeurd

    # voeg de linkerfoto toe op het deel dat niet overlapt
    dst[:h, :s] = left[:h, :s]

    # voeg de foto's samen in het overlappend deel zodat een zachte overgang ontstaat
    step = 1
    for n in range(s, w-step, step):
        x = (n - s) / (w - s)
        dst[:, n] = cv2.addWeighted(left[:, n:n+step], 1 - x, dst[:, n:n+step], x, 0)
    return dst



def eerste_frame(images):
    """
    :param images: tuple met linker- en rechterfoto, is belangrijk!
    :return: berekent de transformatix en de plaats waar overlap start
     deze functie is enkel nodig bij de eerste frame in een video
    """

    left, right = images
    h, w, d = left.shape

    # bereken transformatiematrix, indien mislukt return None,None
    matrix = find_kp_and_matrix((left, right))
    if matrix is None:
        return None, None

    # zoek waar de overlap begint
    dst = cv2.warpPerspective(right, matrix, (2*w, h))
    s = 0
    while (s < w) and not np.sum(dst[:, s]):
        s += 1
    return matrix, s


def stitch_video(left_video, right_video):
    """
    :param left_video: linkervideo = lijst met frames
    :param right_video: rechtervideo = lijst met frames
    :return: panorama video
    """

    # zoek M en s
    keypoints_found = False
    i = -1
    while not keypoints_found:
        i += 1
        M, s = eerste_frame((left_video[i], right_video[i]))
        if M is not None:
            keypoints_found = True

    # stitch alle frames in de video een voor een samen
    pano = []
    for n in range(0, len(left_video)):
        new_frame = stitch_frame((left_video[n], right_video[n]), M, s)
        pano.append(new_frame)
    return pano
