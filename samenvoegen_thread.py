import socket
import struct
import threading
import cv2
import numpy as np
import pickle
from picamera import PiCamera
from picamera.array import PiRGBArray
# from pano_def import *
import time
import stitcher
from queue import LifoQueue
import paramiko

_finish = False
HEADER = 16
# IP_CLIENT = "169.254.186.249" #LODE
IP_CLIENT = "169.254.27.179"
try:
    PORT = 5050

    SERVER = "169.254.186.249"
    ADDR = (SERVER, PORT)
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.close()
    server.bind(ADDR)
except:
    PORT = 5051

    SERVER = socket.gethostbyname(socket.gethostname())
    ADDR = ("169.254.186.249", PORT)
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.close()
    server.bind(ADDR)
print("port:", PORT)
# camera initialization
cam = PiCamera()
cam_res = (320, 240)
cam.resolution = cam_res
cam.framerate = 24
rawCapture = PiRGBArray(cam, size=cam_res)
FRAME_LENGTH = cam_res[0] * cam_res[1] * 3 + 163  # 230563


def run_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("169.254.130.140", username="pi", password="qwertyui")
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python Documents/client_no_length.py")


def handle_client(conn, addr, q):
    global _finish

    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    data = b""
    start = time.time()
    payload_size = struct.calcsize("Q")
    first_frame = True
    while connected:
        end = time.time()
        print('totale tijd:', end - start)
        start = end
        frame2 = q.get()
        data = b""
        # payload_size = struct.calcsize("Q")
        send(conn, "!new_frame")
        # if not size_received:
        #     size_received = True

        # message = conn.recv(HEADER).decode(FORMAT)

        # if message[:len(DISCONNECT_MESSAGE)] == DISCONNECT_MESSAGE:
        #    break

        while len(data) < int(FRAME_LENGTH):
            data += conn.recv(4 * 1024)
        # print(data)
        frame = np.array(pickle.loads(data))
        # merge the two pictures
        if first_frame:
            matrix, s = stitcher.eerste_frame((frame, frame2))
            if matrix is not None:
                first_frame = False
                conn.sendall(pickle.dumps(matrix))
        else:
            pano = stitcher.stitch_frame((frame, frame2), matrix, s)
            cv2.imshow('pano', pano)
        # status, result = stitcher.stitch((frame,frame2))
        # if status == 0:
        #     cv2.imshow('result', result)

        # cv2.imshow("RECEIVING VIDEO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    send(conn, DISCONNECT_MESSAGE)
    # conn.close()
    server.close()
    _finish = True
    # main_thread.join()
    exit(0)


def send(conn, msg):
    message = msg.encode(FORMAT)
    # msg_length = len(message)
    # send_length = str(msg_length).encode(FORMAT)
    # send_length += b' ' * (HEADER - len(send_length))
    # conn.send(send_length)
    conn.send(message)


def video_stream(q):
    time.sleep(0.1)
    for frame2 in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        q.put(frame2.array)
        rawCapture.truncate(0)


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    q = LifoQueue(maxsize=1)
    run_client()
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr, q))
        cameraThread = threading.Thread(target=video_stream, args=(q,))
        cameraThread.start()
        thread.start()
        # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        if _finish:
            thread.join()
            print("stop")
            break


print("[STARTING] server is starting...")
start()
