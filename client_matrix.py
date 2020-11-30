from picamera.array import PiRGBArray
from picamera import PiCamera
import pickle
import numpy as np
import socket

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
cam_res = (320, 240)
camera.resolution = cam_res
camera.framerate = 24
camera.start_preview()
rawCapture = PiRGBArray(camera, size=cam_res)

# setup a connection with the server
HEADER = 16
FORMAT = 'utf-8'
# SERVER = "169.254.186.249"  # LODE
SERVER = "169.254.233.181"  # HEKTOR
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 5050
ADDR = (SERVER, PORT)
client.connect(ADDR)
print(f'[CONNECTED] client is connected with {SERVER}')

print("[STARTING] client is starting...")
LEN_MATRIX = 9
connected = True

# wait for server to ask frame
client.recv(HEADER).decode(FORMAT)

while connected:

    # take a picture
    frame = np.empty((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    camera.capture(frame, 'bgr')

    # turn frame in bytes object and send
    message = pickle.dumps(frame)
    client.sendall(message)

    # receive the matrix
    msg = client.recv(HEADER).decode(FORMAT)
    if msg == "matrix_coming":
        data = b""
        while len(data) < LEN_MATRIX:
            data += client.recv(LEN_MATRIX)
        matrix = np.array(pickle.loads(data))

        # save the matrix on the rpi
        np.savetxt('transformation_matrix.csv', matrix, delimiter=',')
        connected = False
