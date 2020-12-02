from imutils.video import VideoStream
import socket
import pickle
import numpy as np

# initialize the camera
cam_res = (1648, 1232)
camera = VideoStream(resolution=cam_res, framerate=5, usePiCamera=True).start()
while camera.read() is None:
    pass


# setup a connection with the server
HEADER = 16
FORMAT = 'utf-8'
# SERVER = "169.254.186.249"  # LODE
SERVER = "169.254.233.181"  # HEKTOR
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 5050
client.connect((SERVER, PORT))
print(f'[CONNECTED] client is connected with {SERVER}')

print("[STARTING] client is starting...")
LEN_MATRIX = 9
connected = True

# wait for server to ask frame
client.recv(HEADER).decode(FORMAT)

while connected:

    # take a picture
    frame = camera.read()

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
