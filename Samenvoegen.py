import socket
import struct
import threading
import cv2
import numpy as np
import pickle
from picamera import PiCamera
from pano_def import *
import time

HEADER = 64
PORT = 5051
SHAPE = (480, 640, 3)
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = ('192.168.137.128', PORT)
#ADDR = ('192.168.43.140', PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

## initialisations for the camera
# h = 1024 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
# cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
# we need to round to the nearest 16th and 32nd (requirement for picamera)
# cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))
cam_res = (640,480)
# camera initialization
cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    data = b""
    start = time.time()
    payload_size = struct.calcsize("Q")
    # initializations for image capture and stitching
    frame2 = np.empty((cam_res[0], cam_res[1], 3), dtype=np.uint8)
    stitcher = cv2.Stitcher.create()
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
        t1 = time.time()
        
        if message[:len(DISCONNECT_MESSAGE)] == DISCONNECT_MESSAGE:
            break

        while len(data) < int(message):
            data += conn.recv(4 * 1024)
        t2 = time.time()
        print('time between asking and return',t2-t1)
        # print(data)
        frame = np.array(pickle.loads(data))

        # take one picture
        # frame2 = np.empty((cam_res[0], cam_res[1], 3), dtype=np.uint8)  # preallocate image
        cam.capture(frame2, 'bgr')
        print(frame.shape)
        cv2.imshow('ontvangen', frame)
        cv2.imshow('image', frame2)
        cv2.waitKey(2000)

        # convert picture to cv2 format
        # not necessary?

        # merge the two pictures
        print("stitch")
        # result = stitch([frame, frame2])
        status, result = stitcher.stitch((frame,frame2))
        print("end stitch")
        print(status)
        #print(result.shape)
        if status == 0:
            cv2.imwrite('stitched.jpg',result)
            cv2.imshow('result', result)

        # TODO Foto nemen, foto samenvoegen, display
        cv2.imshow("RECEIVING VIDEO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
        print(1)
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()

