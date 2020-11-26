# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
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
    camera = PiCamera()
    cam_res = (320, 240)
    camera.resolution = cam_res
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=cam_res)
    time.sleep(0.1)  # allow camera to warm up

    # load transformation matrix, saved on the pi
    M = np.loadtxt('transformation_matrix.csv', delimiter=',')

    # take pictures and transform them
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        q.put(cv2.warpPerspective(frame.array, M, (2 * cam_res[0], cam_res[1])))
        # q.put(frame.array)
        rawCapture.truncate(0)


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
    thread = threading.Thread(target=stream_video_to_server, args=(q,)) # streams video to server
    cameraThread = threading.Thread(target=video_stream, args=(q,))     # capture video stream
    print('start processes')
    cameraThread.start()
    thread.start()

    # wait for the threads to terminate
    thread.join()
    cameraThread.join()


if __name__ == '__main__':

    # initialize connection
    # SERVER = "169.254.186.249"   # LODE
    SERVER = "169.254.233.181"   # HEKTOR
    PORT = 5555

    print("[STARTING] client is starting...")
    start()
    cv2.destroyAllWindows()
