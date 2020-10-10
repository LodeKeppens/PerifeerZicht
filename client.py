import socket
import struct

import cv2
import numpy as np

cap = cv2.VideoCapture(0)


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
    client.send(msg)


    # message = np.char.encode(msg, encoding=FORMAT)
    # #message = msg.encode(FORMAT)
    # msg_length = len(message)
    # send_length = str(msg_length).encode(FORMAT)
    # print(send_length)
    # send_length += b' ' * (HEADER - len(send_length))
    # client.send(send_length)
    # client.send(message)

while True:
    _, frame = cap.read()
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
        if msg == NEW_FRAME_MESSAGE:
            print(frame.shape)
            if frame is not None:
                send(struct.pack('921600B', *frame.flat))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

send(DISCONNECT_MESSAGE)
# send("Hello World!")
# input()
# send("Hello Everyone!")
# input()
# send("Hello Tim!")
#
# send(DISCONNECT_MESSAGE)