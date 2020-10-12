import socket
import struct
import pickle
import cv2
import numpy as np
import time

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

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

while True:
    frame = camera.capture()
    cv2.imshow("in", frame)
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            print("frame requested")
            a = pickle.dumps(frame)
            message = struct.pack("Q", len(a)) + a
            send(message)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
