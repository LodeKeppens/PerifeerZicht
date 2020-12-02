import cv2
from imutils.video import VideoStream
import time
import stitcher_bw1
import paramiko
import imagezmq
import threading
from queue import Queue


def stitch_and_show(q_server, q_client):
    """
    stitch the two frames and show it
    """
    global finish
    # create window where the video will be displayed
    window_name = "camera"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # make sure we started receiving frames from client
    q_client.get()

    start = time.time()
    n = 0
    while not finish:
        n += 1
        # get frames
        frame_client = q_client.get()
        frame_server = q_server.get()

        # stitch the frames
        stitched = stitcher_bw1.stitch_frame_right_warped((frame_server, frame_client))

        # show image
        cv2.imshow(window_name, stitched)

        # if 'q' is pressed, break from loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            finish = True

    # print average time per frame
    end = time.time()
    print('gemiddeld:', (end-start)/n)


def video_stream(q):
    """
    continuously takes frames, rotates them (camera is upside down) and put them in queue
    """
    global finish
    # camera initialization
    cam_res = (320, 240)
    camera = VideoStream(resolution=cam_res, framerate=40, usePiCamera=True).start()

    # wait for camera to start up
    while camera.read() is None:
        pass

    # wait for stitch process to start
    frame = camera.read()
    q.put(frame)
    while q.full():
        pass

    # take frame, rotate(camera is upside down), and put it in q
    while not finish:
        prev = frame  # little trick to reduce delay between right and left frame
        frame = cv2.rotate(camera.read(), cv2.ROTATE_180)
        q.put(prev)


def video_stream_from_client(q):
    """
    receives the frames from client and puts them in the queue
    """
    global finish
    imageHub = imagezmq.ImageHub()

    # wait for client to run and start sending frames
    imageHub.recv_image()
    imageHub.send_reply()

    # receive frames
    while not finish:
        _, frame = imageHub.recv_image()
        imageHub.send_reply()
        q.put(frame)


def start():
    """
    create queues to communicate between processes
    then creates and starts stitch and camera proces
    """
    # create queues
    q_server = Queue(maxsize=1)
    q_client = Queue(maxsize=1)

    # create and start threads
    stitchThread = threading.Thread(target=stitch_and_show, args=(q_server, q_client))
    cameraThread = threading.Thread(target=video_stream, args=(q_server, ))
    clietThread = threading.Thread(target=video_stream_from_client, args=(q_client, ))
    clietThread.start()
    cameraThread.start()
    stitchThread.start()

    # run client file
    ssh.exec_command("python Documents/client_bw3.py")

    # wait for threads to terminate
    stitchThread.join()
    cameraThread.join()
    clietThread.join()


if __name__ == '__main__':
    finish = False
    # initialize connection
    # IP_CLIENT = "169.254.186.249" #LODE
    IP_CLIENT = "169.254.27.179"  # HEKTOR
    PORT = 5555

    # ssh in the client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")

    print("[STARTING] server is starting...")
    start()

    # stop the client
    ssh.exec_command("sudo pkill python")
