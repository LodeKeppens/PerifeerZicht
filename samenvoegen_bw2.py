import socket
import multiprocessing as mp
import cv2
import numpy as np
import pickle
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import stitcher_bw1
import paramiko
import matplotlib.pyplot as plt


def run_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python Documents/client_thread.py")


def first_frame(conn, q):
    # get frames
    send(conn, "!new_frame")
    frame_server = q.get()
    data = b""
    print('receiving the first frame')
    while len(data) < int(FRAME_LENGTH):
        data += conn.recv(FRAME_LENGTH)
    frame_client = np.array(pickle.loads(data))

    # transform matrix
    matrix = stitcher_bw1.eerste_frame((frame_server, frame_client))
    if matrix is None:
        exit(0)
    print('sending 3x3 transformation matrix')
    print(matrix)
    conn.sendall(pickle.dumps(matrix))


def handle_client(conn, q_server, q_client):
    tijden = {'totaal': [], 'fotos_ophalen': [], 'stitch': [], 'show': []}
    connected = True
    print('first frame')
    first_frame(conn, q_server)
    clietThread = mp.Process(target=video_client, args=(q_client, conn))
    clietThread.start()
    start = time.time()
    while connected:
        end = time.time()
        tijden['totaal'].append(end-start)
        start = end

        # get frames
        frame_server = q_server.get()
        frame_client = q_client.get()
        t1 = time.time()
        tijden['fotos_ophalen'].append(t1-start)

        # merge the pictures
        stitched = stitcher_bw1.stitch_frame_right_warped((frame_server, frame_client))
        t2 = time.time()
        tijden['stitch'].append(t2-t1)
        cv2.imshow('Stitched', stitched)
        t1 = time.time()
        tijden['show'].append(t1-t2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if _FINISH:
            break
    cv2.destroyAllWindows()
    send(conn, DISCONNECT_MESSAGE)
    server.close()
    gemiddeld = sum(tijden['totaal'])/len(tijden['totaal'])
    print(gemiddeld)
    for key in tijden:
        val = tijden[key]
        plt.scatter(range(len(val)), val, label=key)
        print(key, sum(val)/len(val))
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("tijd")
    plt.ylim(0, 0.1)
    plt.show()
    clietThread.terminate()
    clietThread.join()


def send(conn, msg):
    message = msg.encode(FORMAT)
    conn.send(message)


def video_stream(q):
    # camera initialization
    cam = PiCamera()
    cam.resolution = cam_res
    cam.framerate = 24
    rawCapture = PiRGBArray(cam, size=cam_res)
    time.sleep(0.1)

    for frame2 in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        q.put(cv2.rotate(frame2.array, cv2.ROTATE_180))
        rawCapture.truncate(0)


def video_client(q, conn):
    while True:
        data = b""
        send(conn, "!new_frame")
        while len(data) < int(2 * FRAME_LENGTH - 163):
            data += conn.recv(2 * FRAME_LENGTH - 163)
        q.put(np.array(pickle.loads(data)))


def start():
    q_server = mp.Queue(maxsize=1)
    q_client = mp.Queue(maxsize=1)
    print('queues created')
    # run_client()
    thread = mp.Process(target=handle_client, args=(conn, q_server, q_client))
    cameraThread = mp.Process(target=video_stream, args=(q_server,))
    cameraThread.start()
    thread.start()
    # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
    thread.join()
    cameraThread.terminate()
    cameraThread.join()


if __name__ == '__main__':
    _FINISH = False
    _START = False
    # initialize connection
    HEADER = 16
    # IP_CLIENT = "169.254.186.249" #LODE
    # SERVER = "169.254.186.249" #LODE
    IP_CLIENT = "169.254.27.179"  # HEKTOR
    SERVER = "169.254.233.181"  # HEKTOR
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"
    try:
        PORT = 5050
        ADDR = (SERVER, PORT)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(ADDR)
    except:
        PORT = 5051
        ADDR = (SERVER, PORT)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(ADDR)
    print("port:", PORT)
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    conn, _ = server.accept()
    cam_res = (320, 240)
    FRAME_LENGTH = cam_res[0] * cam_res[1] * 3 + 163  # 230563
    print("[STARTING] server is starting...")
    start()
