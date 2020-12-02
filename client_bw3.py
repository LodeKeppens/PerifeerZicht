from imutils.video import VideoStream
import cv2
import time
import numpy as np
import socket
import threading
from queue import Queue
import imagezmq


def video_stream(q):
    """
    :param q: queue
    continuously takes pictures and puts them in q
    """
    # initialize the camera
    cam_res = (320, 240)
    camera = VideoStream(resolution=cam_res, framerate=40, usePiCamera=True).start()
    while camera.read() is None:
        pass

    # load transformation matrix, saved on the pi
    M = np.loadtxt('transformation_matrix.csv', delimiter=',')

    while True:
        q.put(cv2.warpPerspective(camera.read(), M, (2 * cam_res[0], cam_res[1])))


def stream_video_to_server(q):
    """
    :param q: queue
    continuously streams the frames in q to the server
    """
    sender = imagezmq.ImageSender(connect_to=f"tcp://{SERVER}:{PORT}")
    pi_name = socket.gethostname()
    while True:
        sender.send_image(pi_name, q.get())


def start():

    # create queue to communicate between threads
    q = Queue(maxsize=1)

    # create and start the two threads
    thread = threading.Thread(target=stream_video_to_server, args=(q, )) # streams video to server
    cameraThread = threading.Thread(target=video_stream, args=(q, ))     # capture video stream
    print('start processes')
    cameraThread.start()
    thread.start()

    # wait for the threads to terminate, which will never happen (:
    thread.join()
    cameraThread.join()


if __name__ == '__main__':

    # initialize connection
    # SERVER = "169.254.186.249"   # LODE
    SERVER = "169.254.233.181"   # HEKTOR
    PORT = 5555

    print("[STARTING] client is starting...")
    start()
