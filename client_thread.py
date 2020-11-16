# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import socket
import pickle
import matplotlib.pyplot as plt
from queue import LifoQueue
import threading


def send(msg):
    client.sendall(msg)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
cam_res = (320, 240)
camera.resolution = cam_res
camera.framerate = 24
camera.start_preview()
rawCapture = PiRGBArray(camera, size=cam_res)
time.sleep(0.1)


def video_stream(q,M):
    """
    :param q: queue
    continiously takes pictures and puts them in q
    """
    print('video stream')
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        q.put(cv2.warpPerspective(frame.array, M, (2 * cam_res[0], cam_res[1])))
        rawCapture.truncate(0)
        if _FINISH:
            break


def first_frame():
    """
    :param q: queue
    first sends a frame to the server, then server sends back the transformation matrix
    """
    LEN_MATRIX = 9
    msg = client.recv(HEADER).decode(FORMAT)
    frame = np.empty((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    camera.capture(frame, 'bgr')
    print('send message')
    message = pickle.dumps(frame) # Turns the image into a bytes object.
    send(message)
    print(len(message))
    data = b""
    while len(data) < LEN_MATRIX:
        data += client.recv(4 * 1024)
    matrix = np.array(pickle.loads(data))
    return matrix


def handle_server(q):
    connected = True
    tijden = {'totaal': [], 'foto_nemen': [], 'wachten_op_vraag': [], 'send': [], 'transformatie': []}
    start = time.time()
    t1 = start
    n = 0
    while connected:
        # if n > 100:
        #     break
        end = time.time()
        print(end-start)
        tijden['totaal'].append(end - start)
        if time.time() - t1 > 3:
            exit(0)
        start = end
        msg = client.recv(HEADER).decode(FORMAT)  # Turns the incoming message
        # from a bytes object to an string.
        if msg == NEW_FRAME_MESSAGE:
            t1 = time.time()
            tijden['wachten_op_vraag'].append(t1 - start)
            message = pickle.dumps(q.get())  # Turns the image into a bytes object.
            t3 = time.time()
            tijden['foto_nemen'].append(t3 - t1)
            client.sendall(message)
            t2 = time.time()
            tijden['send'].append(t2 - t3)
            n += 1
        elif msg == DISCONNECT_MESSAGE:
            break

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # for key in tijden:
    #     plt.scatter(range(len(tijden[key])), tijden[key], label=key)
    # plt.legend()
    # plt.xlabel("frame")
    # plt.ylabel("tijd")
    # plt.ylim(0, 0.1)
    # plt.show()
    exit(0)


def start():
    global _FINISH
    print('first frame')
    matrix = first_frame()
    q = LifoQueue(maxsize=1)
    thread = threading.Thread(target=handle_server, args=(q,))
    cameraThread = threading.Thread(target=video_stream, args=(q, matrix))
    print('start processes')
    cameraThread.start()
    thread.start()
    thread.join()
    _FINISH = True
    cameraThread.join()


if __name__ == '__main__':
    _FINISH = False
    # setup a connection with the server
    HEADER = 16
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"
    NEW_FRAME_MESSAGE = "!new_frame"
    # SERVER = "169.254.186.249" #LODE
    SERVER = "169.254.233.181"  # HEKTOR
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        PORT = 5050
        ADDR = (SERVER, PORT)
        client.connect(ADDR)
    except:
        PORT = 5051
        ADDR = (SERVER, PORT)
        client.connect(ADDR)
    print(f'[CONNECTED] client is connected with {SERVER}')

    print("[STARTING] client is starting...")
    start()
    # disc_msg = DISCONNECT_MESSAGE
    # send_disc_msg = str(disc_msg).encode(FORMAT)
    # send_disc_msg += b' ' * (HEADER - len(send_disc_msg))
    # client.send(send_disc_msg)
    cv2.destroyAllWindows()
