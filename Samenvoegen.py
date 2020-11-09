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
import video
import queue

_finish = False
HEADER = 16
try:
    PORT = 5050

    SERVER = socket.gethostbyname(socket.gethostname())
    ADDR = ("169.254.186.249", PORT)
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
cam.framerate = 32
rawCapture = PiRGBArray(cam, size=cam_res)


## initialisations for the camera
# h = 1024 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
# cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
# we need to round to the nearest 16th and 32nd (requirement for picamera)
# cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))


def handle_client(conn, addr):
    global _finish

    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    data = b""
    start = time.time()
    payload_size = struct.calcsize("Q")
    # initializations for image capture and stitching
    frame2 = np.empty((cam_res[0], cam_res[1], 3), dtype=np.uint8)
    stitcher = cv2.Stitcher.create()
    first_frame = True
    while connected:
        time.sleep(0.1)
        for frame2 in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame2 = frame2.array
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
            message = 230563

            while len(data) < int(message):
                data += conn.recv(4 * 1024)
            # print(data)
            frame = np.array(pickle.loads(data))
            t1 = time.time()
            print("received data", t1 - start)
            # merge the two pictures
            if first_frame:
                t2 = time.time()
                first_frame = False
                print('matrix berekenen')
                # matrix, s = video.eerste_frame((frame, frame2))
                matrix, _ = video.find_kp_and_matrix((frame, frame2))
                s = 0
                t3 = time.time()
                print("matrix berekenen", t3 - t2)
            if not first_frame:
                t4 = time.time()
                pano = video.match_pano((frame, frame2), matrix, s)
                t5 = time.time()
                print("stitching", t5 - t4)
                cv2.imshow('pano', pano)
            # status, result = stitcher.stitch((frame,frame2))
            # if status == 0:
            #     cv2.imshow('result', result)

            # TODO Foto nemen, foto samenvoegen, display
            # cv2.imshow("RECEIVING VIDEO", frame)
            rawCapture.truncate(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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


def video_stream():


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        print(1)
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        if _finish:
            thread.join()
            print("stop")
            break


print("[STARTING] server is starting...")
start()