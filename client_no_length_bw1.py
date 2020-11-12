# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import socket
import pickle
import matplotlib.pyplot as plt

# import struct


HEADER = 11
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
NEW_FRAME_MESSAGE = "!new_frame"
# SERVER = "169.254.186.249" #LODE
SERVER = "169.254.233.181" #HEKTOR
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    PORT = 5050
    ADDR = (SERVER, PORT)
    client.connect(ADDR)
except:
    PORT = 5051
    ADDR = (SERVER, PORT)
    client.connect(ADDR)


def send(msg):
    client.sendall(msg)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
cam_res = (320, 240)
camera.resolution = cam_res
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=cam_res)
# allow the camera to warmup
time.sleep(0.1)

is_first_frame = True
LEN_MATRIX = 9
# capture frames from the camera
tijden = {'totaal':[],'foto_nemen':[],'wachten_op_vraag':[],'send':[],'transformatie':[]}
start = time.time()
t2 = start
t1 = start
n=0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # if n > 100:
    #     break
    end = time.time()
    tijden['totaal'].append(end-start)
    tijden['foto_nemen'].append(end-t2)
    if time.time()-t1 > 10:
        exit(0)
    start = end
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # save image
    msg = client.recv(HEADER).decode(FORMAT)  # Turns the incoming message
    # from a bytes object to an string.
    if msg == NEW_FRAME_MESSAGE:
        t1 = time.time()
        tijden['wachten_op_vraag'].append(t1-start)
        if is_first_frame:
            message = pickle.dumps(image)  # Turns the image into a bytes object.
            client.sendall(message)

            data = b""
            while len(data) < LEN_MATRIX:
                data += client.recv(4 * 1024)
            matrix = np.array(pickle.loads(data))
            is_first_frame = False
            tijd_eerste_frame = time.time()-t1
        else:
            h, w, d = image.shape
            message = cv2.warpPerspective(image, matrix, (2 * w, h))
            t2 = time.time()
            tijden['transformatie'].append(t2-t1)
            message = pickle.dumps(message)  # Turns the image into a bytes object.
            client.sendall(message)
            t1 = time.time()
            tijden['send'].append(t1-t2)
            n+=1

        print(len(message))
    elif msg == DISCONNECT_MESSAGE:
        exit(0)
    # show the frame
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    t2 = time.time()

for key in tijden:
    plt.scatter(range(len(tijden[key])), tijden[key], label=key)
plt.legend()
plt.xlabel("frame")
plt.ylabel("tijd")
plt.ylim(0, 0.12)
# plt.xlim(0, len(x))
plt.show()

disc_msg = DISCONNECT_MESSAGE
send_disc_msg = str(disc_msg).encode(FORMAT)
send_disc_msg += b' ' * (HEADER - len(send_disc_msg))
client.send(send_disc_msg)
cv2.destroyAllWindows()