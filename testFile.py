import time
import cv2
import numpy as np
import socket
import pickle
import struct

camera = cv2.VideoCapture(0)
HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
SERVER = "192.168.56.1"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    client.sendall(msg)

# initialize the camera and grab a reference to the raw camera capture
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
while True:
    _, image = camera.read()
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    # save image
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            print("frame requested")
            message = pickle.dumps(image)
            msg_length = len(message)
            send_length = str(msg_length).encode(FORMAT)
            send_length += b' ' * (HEADER - len(send_length))
            client.send(send_length)
            client.send(message)
            # a = pickle.dumps(image)
            # message = a
            # send(message)
    # show the frame
    #cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()