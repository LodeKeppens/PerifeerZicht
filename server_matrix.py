import socket
import cv2
import numpy as np
import pickle
from picamera import PiCamera
from picamera.array import PiRGBArray
import stitcher_bw1
import paramiko


def run_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python Documents/client_matrix.py")


# camera initialization
cam = PiCamera()
cam_res = (320, 240)
cam.resolution = cam_res
cam.framerate = 24
rawCapture = PiRGBArray(cam, size=cam_res)

# initialize connection
HEADER = 16
# IP_CLIENT = "169.254.186.249" #LODE
# SERVER = "169.254.186.249" #LODE
IP_CLIENT = "169.254.27.179"  # HEKTOR
SERVER = "169.254.233.181"  # HEKTOR
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
PORT = 5050
ADDR = (SERVER, PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"[LISTENING] Server is listening on {SERVER}")

run_client()
conn, _ = server.accept()
FRAME_LENGTH = cam_res[0] * cam_res[1] * 3 + 163  # 230563
print("[STARTING] server is starting...")
connected = True
while connected:

    # ask new frame
    message = 'new_frame!'
    message = message.encode(FORMAT)
    conn.send(message)

    # take a picture
    frame_server = np.empty((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    cam.capture(frame_server, 'bgr')

    # receive picture from client
    data = b""
    while len(data) < int(FRAME_LENGTH):
        data += conn.recv(FRAME_LENGTH)
    frame_client = np.array(pickle.loads(data))

    # find the transformation matrix
    matrix = stitcher_bw1.find_kp_and_matrix((frame_server, frame_client))

    # show panorama and check if we found a good matrix
    cv2.imshow('panorama', stitcher_bw1.stitch_frame((frame_server, frame_client), matrix))
    answer = input("enter True if panorama success, False to try again: ")
    if answer == "True":
        # send matrix back to client
        message = 'matrix_coming'
        message = message.encode(FORMAT)
        conn.send(message)
        conn.sendall(pickle.dumps(matrix))
        connected = False
