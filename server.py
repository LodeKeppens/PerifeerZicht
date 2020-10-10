import socket
import struct
import threading
import cv2
import numpy as np

HEADER = 64
PORT = 5050
SHAPE = (480, 640, 3)
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        send(conn, "!new_frame")

        data = conn.recv(921600)  # read 230400 bytes
        arr = np.array(struct.unpack('921600B', data), dtype='B').reshape((640, 480, 3), )
        cv2.imshow("output", arr)

        # msg_length = conn.recv(HEADER).decode(FORMAT)
        # if msg_length:
        #     msg_length = int(msg_length)
        #     print(msg_length)
        #     msg = conn.recv(msg_length).decode(FORMAT)
        #     msg = str2ndarray(msg)
        #     if msg == DISCONNECT_MESSAGE:
        #         connected = False
        #
        #     print("image")
        #     cv2.imshow("stream", msg)
            #print(f"[{addr}] {msg}")

    conn.close()

def send(conn, msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    conn.send(send_length)
    conn.send(message)

def str2ndarray(a):
    # Specify your data type, mine is numpy float64 type, so I am specifying it as np.float64
    a = np.fromstring(a, dtype=int)
    a = np.reshape(a, SHAPE)

    return a

def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()