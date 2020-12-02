import multiprocessing as mp
import cv2
from imutils.video import VideoStream
import time
import stitcher_bw1
import paramiko
import imagezmq
import matplotlib.pyplot as plt


def stitch_and_show(q_server, q_client):
    """
    stitch the two frames and show it
    """

    # create window where the video will be displayed
    window_name = "camera"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # make sure we start recieving frames from client
    q_client.get()

    start = time.time()
    t1 = start
    nb_frames = 0
    while True:
        t0 = t1
        t1 = time.time()
        nb_frames += 1

        # get frames
        frame_client = q_client.get()
        frame_server = q_server.get()
        t2 = time.time()

        # stitch the frames
        stitched = stitcher_bw1.stitch_frame_right_warped((frame_server, frame_client))
        t3 = time.time()

        # show image
        cv2.imshow(window_name, stitched)
        t4 = time.time()

        # if 'q' is pressed, break from loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # print average time per frame
    end = time.time()
    print('gemiddeld:', (end-start)/nb_frames)


def video_stream(q):
    # camera initialization
    cam_res = (320, 240)
    camera = VideoStream(resolution=cam_res, framerate=40, usePiCamera=True).start()
    # wait for camera to start up
    while camera.read() is None:
        pass

    # take frame, rotate(camera is upside down), and put it in q
    frame = camera.read()
    while True:
        prev = frame  # little trick to reduce delay between right and left frame
        frame = cv2.rotate(camera.read(), cv2.ROTATE_180)
        q.put(prev)


def video_stream_from_client(q):
    """
    :param q: qeueu where recieved frames will come
    recieves the frames from client
    """
    imageHub = imagezmq.ImageHub()
    while True:
        _, frame = imageHub.recv_image()
        imageHub.send_reply()
        q.put(frame)


def start():
    """
    create queues to communicate between processes
    then creates and starts stitch and camera proces
    """
    # create queues
    q_server = mp.Queue(maxsize=1)
    q_client = mp.Queue(maxsize=1)

    # create and start processes
    stitchProcess = mp.Process(target=stitch_and_show, args=(q_server, q_client))
    cameraProcess = mp.Process(target=video_stream, args=(q_server, ))
    clietProcess = mp.Process(target=video_stream_from_client, args=(q_client, ))
    clietProcess.start()
    cameraProcess.start()
    stitchProcess.start()

    # wait for stitchProcess to terminate, then terminate the other processes
    stitchProcess.join()
    cameraProcess.terminate()
    clietProcess.terminate()
    cameraProcess.join()
    clietProcess.join()


if __name__ == '__main__':

    # initialize connection
    # IP_CLIENT = "169.254.186.249" #LODE
    IP_CLIENT = "169.254.27.179"  # HEKTOR
    PORT = 5555

    # ssh in the client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")

    # run client file
    ssh.exec_command("python Documents/client_bw3.py")

    print("[STARTING] server is starting...")
    start()

    # stop the client
    ssh.exec_command("sudo pkill python")
