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
import stitcher_bw1
from queue import LifoQueue
import paramiko

_finish = False
HEADER = 16
# IP_CLIENT = "169.254.186.249" #LODE
# SERVER = "169.254.186.249" #LODE
IP_CLIENT = "169.254.27.179" #HEKTOR
SERVER = "169.254.233.181" #HEKTOR
try:
    PORT = 5050

    ADDR = (SERVER, PORT)
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.close()
    server.bind(ADDR)
except:
    PORT = 5051

    #SERVER = socket.gethostbyname(socket.gethostname())
    ADDR = (SERVER, PORT)
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
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python Documents/client_no_length_bw1.py")


def handle_client(conn, addr, q):
    global _finish

    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    data = b""
    start = time.time()
    payload_size = struct.calcsize("Q")
    is_first_frame = True
    while connected:
        frame_server = q.get()
        frame_server = cv2.rotate(frame_server, cv2.ROTATE_180) #ENKEL_VOOR_OPSTELLING_HEKTOR
        end = time.time()
        print('totale tijd:', end - start)
        start = end

        data = b""
        # payload_size = struct.calcsize("Q")
        send(conn, "!new_frame")
        # if not size_received:
        #     size_received = True

        # message = conn.recv(HEADER).decode(FORMAT)

        # if message[:len(DISCONNECT_MESSAGE)] == DISCONNECT_MESSAGE:
        #    break


        # merge the two pictures
        if is_first_frame:
            print('receiving the first frame')
            while len(data) < int(FRAME_LENGTH):
                data += conn.recv(4 * 1024)
            frame_client = np.array(pickle.loads(data))

            matrix, s = stitcher_bw1.eerste_frame((frame_server, frame_client))
            if matrix is not None:
                is_first_frame = False
                print('sending 3x3 transformation matrix')
                conn.sendall(pickle.dumps(matrix))
        else:
            print('receiving the next frame')
            while len(data) < int(2*FRAME_LENGTH):
                data += conn.recv(4 * 1024)
            frame_client = np.array(pickle.loads(data))

            print('beginning to stitch')
            stitched = stitcher_bw1.stitch_frame_right_warped((frame_server, frame_client), matrix, s)
            cv2.imshow('Stitched', stitched)
        # status, result = stitcher.stitch((frame_client,frame_server))
        # if status == 0:
        #     cv2.imshow('result', result)

        # cv2.imshow("RECEIVING VIDEO", frame_client)
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
