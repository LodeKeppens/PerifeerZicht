import socket
import struct
import threading
import cv2
import numpy as np
import pickle

HEADER = 64
PORT = 5050
SHAPE = (480, 640, 3)
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

## initialisations for the camera
# h = 1024 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
# cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
# we need to round to the nearest 16th and 32nd (requirement for picamera)
# cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))
cam_res = (640,480)
# camera initialization
cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    data = b""
    payload_size = struct.calcsize("Q")
    frame2 = np.empty((cam_res[0], cam_res[1], 3), dtype=np.uint8)  # preallocate image
    stitcher = cv2.Stitcher.create()

    while connected:
        send(conn, "!new_frame")

        while len(data) < payload_size:
            packet = conn.recv(4 * 1024)  # 4K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = np.array(pickle.loads(frame_data))
        print(frame)

        # take one picture
        # frame2 = np.empty((cam_res[0], cam_res[1], 3), dtype=np.uint8)  # preallocate image
        cam.capture(frame2, 'rgb')

        # convert picture to cv2 format


        # merge the two pictures
        images = [frame, frame2]
        (status, result) = stitcher.stitch(images)
        if status == cv2.STITCHER_OK:
            print('panorama generated')
            cv2.imshow('result', result)

        # TODO Foto nemen, foto samenvoegen, display

        cv2.imshow("RECEIVING VIDEO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    conn.close()

def send(conn, msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    conn.send(send_length)
    conn.send(message)

def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()

