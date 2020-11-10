import time
import cv2
import numpy as np
import socket
import pickle
import struct


camera = cv2.VideoCapture(0)
size_send = False
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
    print("send")

# initialize the camera and grab a reference to the raw camera capture
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
while True:
    _, image = camera.read()

    #cv2.imshow("Frame", image)
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    # save image

    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            message = pickle.dumps(image)
            if not size_send:
                size_send = True
                msg_length = len(message)
                print(msg_length)
                send_length = str(msg_length).encode(FORMAT)
                send_length += b' ' * (HEADER - len(send_length))
                client.send(send_length)
            client.send(message)
    # show the frame
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
