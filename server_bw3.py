import multiprocessing as mp
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import stitcher_bw1
import paramiko
import imagezmq


def run_client():
    """
    runs the client trough ssh, so the user only needs to run the server
    """

    # ssh in the client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(IP_CLIENT, username="pi", password="qwertyui")

    # run the client file
    # !make sure filename corresponds with the one on the client!
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python Documents/client_bw3.py")


def stitch_and_show(q_server, q_client):
    # start display process (we created an extra process because the resizing is very slow)
    q_video = mp.Queue(maxsize=1)
    display = mp.Process(target=show, args=(q_video,))
    display.start()

    start = time.time()
    nb_frames = 0
    while True:
        nb_frames += 1
        # get frames
        frame_client = q_client.get()
        frame_server = q_server.get()

        # stitch the frames
        stitched = stitcher_bw1.stitch_frame_right_warped((frame_server, frame_client))

        # if display has terminated, break the loop
        if display.exitcode is not None:
            break

        # show image
        # cv2.imshow('video', stitched)
        q_video.put(stitched)

    # print average time per frame
    end = time.time()
    print((end-start)/nb_frames)
    display.terminate()
    display.join()


def video_stream(q):
    # camera initialization
    cam = PiCamera()
    cam.resolution = cam_res
    cam.framerate = 24
    rawCapture = PiRGBArray(cam, size=cam_res)
    time.sleep(0.1)

    # take frame, rotate(camera is upside down), and put it in q
    for frame2 in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        q.put(cv2.rotate(frame2.array, cv2.ROTATE_180))
        rawCapture.truncate(0)


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


def show(q):
    """
    :param q: queue with the stitched frames
    shows the live video
    """

    # create named window where video will be showed
    window_name = "camera"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # get stitched frame and show it
    while True:
        # frame = imutils.resize(q.get(), width=640, height=480)
        frame = q.get()
        cv2.imshow(window_name, frame)

        # if 'q' is pressed, break from loop and close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


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
    cameraProcess = mp.Process(target=video_stream, args=(q_server,))
    clietProcess = mp.Process(target=video_stream_from_client, args=(q_client,))
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
    # SERVER = "169.254.186.249"    #LODE
    IP_CLIENT = "169.254.27.179"  # HEKTOR
    SERVER = "169.254.233.181"    # HEKTOR

    # run_client() # hier is nog probleempje mee, client runt wel degelijk ,(want krijg pycameraerror bij 2e poging)

    cam_res = (320, 240)
    FRAME_LENGTH = cam_res[0] * cam_res[1] * 3 + 163  # 230563
    print("[STARTING] server is starting...")
    start()
