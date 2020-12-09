import numpy as np
import cv2


hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)
frames_list = []
resized = None
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
    # found_filtered = []
    # for ri, r in enumerate(found):
    #     for qi, q in enumerate(found):
    #         if ri != qi and inside(r, q):
    #             break
    #     else:
    #         found_filtered.append(r)
    # for x, y, w, h in found:
    #     # the HOG detector returns slightly larger rectangles than the real objects.
    #     # so we slightly shrink the rectangles to get a nicer output.
    #     pad_w, pad_h = int(0.15*w), int(0.05*h)
    #     cv2.rectangle(frame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 1)
    #draw_detections(frame, found_filtered, 3)
    #print('%d (%d) found' % (len(found_filtered), len(found)))
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()