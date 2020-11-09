# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import socket
import pickle

# import struct


HEADER = 11
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
SERVER = "169.254.186.249"
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    PORT = 5050
    ADDR = (SERVER, PORT)
    client.connect(ADDR)
except:
    PORT = 5051
    ADDR = (SERVER, PORT)
    client.connect(ADDR)


def send(msg):
    client.sendall(msg)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
cam_res = (320, 240)
camera.resolution = cam_res
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=cam_res)
# allow the camera to warmup
time.sleep(0.1)

is_first_frame = True
LEN_MATRIX = 9
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # save image
    msg = client.recv(HEADER).decode(FORMAT)  # Turns the incoming message
    # from a bytes object to an string.
    if msg == NEW_FRAME_MESSAGE:
        if not is_first_frame:
            message = pickle.dumps(image)  # Turns the image into a bytes object.
            client.sendall(message)

            data = b""
            while len(data) < LEN_MATRIX:
                data += client.recv(4 * 1024)
            matrix = np.array(pickle.loads(data))
        else:


        print(len(message))
    elif msg == DISCONNECT_MESSAGE:
        exit(0)
    # show the frame
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop

disc_msg = DISCONNECT_MESSAGE
send_disc_msg = str(disc_msg).encode(FORMAT)
send_disc_msg += b' ' * (HEADER - len(send_disc_msg))
client.send(send_disc_msg)
cv2.destroyAllWindows()