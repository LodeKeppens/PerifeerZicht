import cv2
import os
import time

images = []
path = "fotos"
myList = os.listdir(path)
print(myList)

# Het tonen van de verschillende foto's achter elkaar
# Foto's in een lijst plaatsen
for img in myList:
    curImg = cv2.imread(f'{path}/{img}')
    curImg = cv2.resize(curImg, (640, 480))
    images.append(curImg)
    # Naam van de image aanduiden: f'image...
    cv2.imshow(f'image{len(images)}', curImg)
    cv2.waitKey(500)

stitcher = cv2.Stitcher.create()
# Tijdsduur opmeten
start = time.time()
# Het effectieve stitchen
# Status: of het al dan niet is gelukt; result: de panorama
(status, result) = stitcher.stitch(images)
end = time.time()
print(end - start)
if status == cv2.STITCHER_OK:
    print('panorama generated')
    cv2.imshow('result', result)
    print(result.shape)
    # Het bijsnijden van de foto's
    pan = result[40:400, 50:790]
    cv2.imshow('panorama', pan)
else:
    print('unsuccessfull')
cv2.waitKey(0)
