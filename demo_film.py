import cv2
import video
import stitcher_bw1
import numpy as np


def stitchh(left,right):
    stitcher = cv2.Stitcher.create()
    stitched = stitcher.stitch(left, right)
    print(stitched[0])
    return stitched[1]


# Read the images
left = cv2.resize(cv2.imread('fotos/left2.jpg'), (320, 240))
right = cv2.resize(cv2.imread('fotos/right2.jpg'), (320, 240))

# create the right matrix
h, w, d = 240, 560, 3
dst = np.zeros((h, w, d), dtype='uint8')
dst[:, w-320:] = right

dst = stitcher_bw1.stitch_frame_right_warped((left, dst))
print(dst.shape)
cv2.imwrite('fotos/unwarped_result.jpg',dst)
# dst2 = stitchh(left,right)

cv2.imshow('result', dst)
# cv2.imshow('result2', dst2)
cv2.waitKey(000)


cv2.destroyAllWindows()

