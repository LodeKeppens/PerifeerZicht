# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import socket
import pickle
import struct


HEADER = 64
PORT = 5051
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
#SERVER = "192.168.137.128"
SERVER = "169.254.233.181"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    client.sendall(msg)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 64
time.sleep(0.1)
image = np.empty((480,640,3), dtype=np.uint8)

while True:
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        print('frame asked')
        if msg == NEW_FRAME_MESSAGE:
            t1 = time.time()
            camera.capture(image, 'bgr')
            print('foto nemen', t1-time.time())
            # cv2.imshow('frame', image)
            message = pickle.dumps(image)
            msg_length = len(message)
            send_length = str(msg_length).encode(FORMAT)
            send_length += b' ' * (HEADER - len(send_length))
            client.send(send_length)
            client.send(message)
        elif msg == DISCONNECT_MESSAGE:
            exit(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    
disc_msg = DISCONNECT_MESSAGE
send_disc_msg = str(disc_msg).encode(FORMAT)
send_disc_msg += b' ' * (HEADER - len(send_disc_msg))
client.send(send_disc_msg)
cv2.destroyAllWindows()
