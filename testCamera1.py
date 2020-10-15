# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
from SSH_test import *

source = r'/home/pi/Documents/frame.npy'
dest = r'Documents/frame.npy'
# initialize the camera and grab a reference to the raw camera capture
h = 1024 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
# we need to round to the nearest 16th and 32nd (requirement for picamera)
cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))
# camera initialization
cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])
data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8) # preallocate image
cam.capture(data,'rgb')
#start = time.time()
# capture frames from the camera

# grab the raw NumPy array representing the image, then initialize the timestamp
# and occupied/unoccupied text
# save image
np.save('frame',data)
# send image to the other PI
sendData(source, dest)
exit(0)



