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
import matplotlib.pyplot as plt



def video_stream(q, q_times_video):
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
    times = []
    time_start = time.time()
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        time_end = time.time()
        times.append(time_end - time_start)
        q_times_video.put(times)
        time_start = time_end

        q.put(cv2.warpPerspective(frame.array, M, (2 * cam_res[0], cam_res[1])))
        # q.put(frame.array)
        rawCapture.truncate(0)


def stream_video_to_server(q,q_times_to_server):
    """
    :param q: queue
    continuously streams the frames in q to the server
    """
    sender = imagezmq.ImageSender(connect_to=f"tcp://{SERVER}:{PORT}")
    pi_name = socket.gethostname()
    times = []
    time_start = time.time()
    while True:
        time_end = time.time()
        times.append(time_end - time_start)
        q_times_to_server.put(times)
        time_start = time_end

        sender.send_image(pi_name, q.get())


def start():

    # create queue to communicate between threads
    q = Queue(maxsize=1)
    q_times_video = Queue(maxsize=1)
    q_times_to_server = Queue(maxsize=1)

    # create and start the two threads
    thread = threading.Thread(target=stream_video_to_server, args=(q, q_times_to_server)) # streams video to server
    cameraThread = threading.Thread(target=video_stream, args=(q, q_times_video))     # capture video stream
    print('start processes')
    cameraThread.start()
    thread.start()

    # wait for the threads to terminate
    thread.join()
    cameraThread.join()

    tijden = {'video':q_times_video.get(), 'to server':q_times_to_server.get()}
    for key in tijden:
        plt.scatter(range(len(tijden[key])), tijden[key], label=key)
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("tijd")
    plt.ylim(0, 0.12)
    # plt.xlim(0, len(x))
    plt.show()


if __name__ == '__main__':

    # initialize connection
    # SERVER = "169.254.186.249"   # LODE
    SERVER = "169.254.233.181"   # HEKTOR
    PORT = 5555

    print("[STARTING] client is starting...")
    start()
    cv2.destroyAllWindows()
