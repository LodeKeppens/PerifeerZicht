# import socket
# import struct
# import pickle
# import cv2
# import numpy as np
# import time
# import io
#
# from picamera.array import PiRGBArray
# import picamera.array
# import picamera
# from picamera import PiCamera
#
# camera = PiCamera()
# camera.resolution = (640, 480)
# #rawCapture = PiRGBArray(camera, size=(640, 480))
# time.sleep(0.1)
#
# HEADER = 64
# PORT = 5050
# FORMAT = 'utf-8'
# DISCONNECT_MESSAGE = "!DISCONNECT"
# NEW_FRAME_MESSAGE = "!new_frame"
# SERVER = "169.254.136.56"
# ADDR = (SERVER, PORT)
#
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(ADDR)
#
# def send(msg):
#     client.sendall(msg)
#
#
# frame = np.zeros((640,480,3))
# with picamera.PiCamera() as camera:
#     camera.start_preview()
#     time.sleep(2)
#     while True:  # Create the in-memory stream
#         camera.capture(frame, format='jpeg')
#         # Construct a numpy array from the stream
#         data = np.fromstring(frame.getvalue(), dtype=np.uint8)
#         # "Decode" the image from the array, preserving colour
#         image = cv2.imdecode(data, 1)
#         # OpenCV returns an array with data in BGR order. If you want RGB instead
#         # use the following...
#         frame = image[:, :, ::-1]
#
#         cv2.imshow("in", frame)
#         # msg_length = client.recv(HEADER).decode(FORMAT)
#         # if msg_length:
#         #     msg_length = int(msg_length)
#         #     msg = client.recv(msg_length).decode(FORMAT)
#         #     if msg == NEW_FRAME_MESSAGE:
#         #         print("frame requested")
#         #         a = pickle.dumps(frame)
#         #         message = struct.pack("Q", len(a)) + a
#         #         send(message)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#


# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import socket
import pickle
import struct


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
SERVER = "169.254.136.56"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    client.sendall(msg)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
start = time.time()
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = np.array(frame)
    # save image
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            print("frame requested")
            a = pickle.dumps(image)
            message = struct.pack("Q", len(a)) + a
            send(message)
    # show the frame
    #cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()