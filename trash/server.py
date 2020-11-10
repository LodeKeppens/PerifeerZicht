import socket
import struct
import threading
import cv2
import numpy as np
import pickle
import time

HEADER = 64
PORT = 5050
SHAPE = (480, 640, 3)
SERVER = socket.gethostbyname(socket.gethostname())
# ADDR = (SERVER, PORT)
ADDR = ('192.168.43.140', PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    message = None
    # size_received = False
    connected = True
    start = time.time()
    while connected:
        end = time.time()
        print(end-start)
        start = end

        data = b""
        # payload_size = struct.calcsize("Q")
        send(conn, "!new_frame")
        # if not size_received:
        #     size_received = True

        message = conn.recv(HEADER).decode(FORMAT)
        if message[:len(DISCONNECT_MESSAGE)] == DISCONNECT_MESSAGE:
            break

        while len(data) < int(message):
            data += conn.recv(4 * 1024)
        # print(data)
        frame = np.array(pickle.loads(data))
        # print(frame)
        cv2.imshow("RECEIVING VIDEO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    send(conn,DISCONNECT_MESSAGE)
    cv2.destroyAllWindows()
    conn.close()

def send(conn, msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    conn.send(send_length)
    conn.send(message)

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
