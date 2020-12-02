import socket
import cv2
import numpy as np
import pickle
from imutils.video import VideoStream
import stitcher_bw1
import paramiko

show_res = (640, 480)
# camera initialization
cam_res = (1648, 1232)
camera = VideoStream(resolution=cam_res, framerate=5, usePiCamera=True).start()
while camera.read() is None:
    pass

# initialize connection
HEADER = 16
FORMAT = 'utf-8'
# IP_CLIENT = "169.254.186.249" #LODE
# SERVER = "169.254.186.249" #LODE
IP_CLIENT = "169.254.27.179"  # HEKTOR
SERVER = "169.254.233.181"  # HEKTOR
PORT = 5050
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((SERVER, PORT))
server.listen()
print(f"[LISTENING] Server is listening on {SERVER}")

# run client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
ssh.connect(IP_CLIENT, username="pi", password="qwertyui")
ssh.exec_command("python Documents/client_matrix.py")

conn, _ = server.accept()
FRAME_LENGTH = cam_res[0] * cam_res[1] * 3 + 163
print("[STARTING] server is starting...")
connected = True
while connected:

    # ask new frame
    message = 'new_frame!'
    message = message.encode(FORMAT)
    conn.send(message)

    # take a picture
    frame_server = cv2.rotate(camera.read(), cv2.ROTATE_180)

    # receive picture from client
    data = b""
    while len(data) < int(FRAME_LENGTH):
        data += conn.recv(FRAME_LENGTH)
    frame_client = np.array(pickle.loads(data))

    # find the transformation matrix
    print("computing transformation matrix...")
    matrix = stitcher_bw1.find_kp_and_matrix((frame_server, frame_client))

    if matrix is not None:
        # show panorama
        print("stitching frames, to show result...")
        stitched = stitcher_bw1.stitch_frame((frame_server, frame_client), matrix)
        cv2.imshow('server', cv2.resize(frame_server, show_res))
        cv2.imshow('client', cv2.resize(frame_client, show_res))
        cv2.imshow('panorama', cv2.resize(stitched, (show_res[0]*2, show_res[1])))
        cv2.waitKey(0)
        # and ask if the result is good
        answer = input("try again?(yes/no): ")
        if answer == "no":
            # send matrix back to client
            message = 'matrix_coming'
            message = message.encode(FORMAT)
            conn.send(message)
            conn.sendall(pickle.dumps(matrix))
            conn.close()
            connected = False
            print('matrix send to client')

    else:
        print('not enough keypoints found, trying again...')
