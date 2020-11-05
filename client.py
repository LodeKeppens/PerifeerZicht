# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import socket
import pickle

# import struct


HEADER = 64
PORT = 5051
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
SERVER = "169.254.186.249"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)


def send(msg):
    client.sendall(msg)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # save image
    msg_length = client.recv(HEADER).decode(FORMAT)  # Turns the incoming message
    # from a bytes object to an string.
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            message = pickle.dumps(image)  # Turns the image into a bytes object.
            msg_length = len(message)
            send_length = str(msg_length).encode(FORMAT)
            send_length += b' ' * (HEADER - len(send_length))
            client.sendall(send_length)
            client.sendall(message)
        elif msg == DISCONNECT_MESSAGE:
            exit(0)
    # show the frame
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

disc_msg = DISCONNECT_MESSAGE
send_disc_msg = str(disc_msg).encode(FORMAT)
send_disc_msg += b' ' * (HEADER - len(send_disc_msg))
client.send(send_disc_msg)
cv2.destroyAllWindows()