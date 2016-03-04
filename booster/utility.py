"""
Utility functions for motion field transfer.

Stuff like playing and reading videos.
"""

import numpy as np
import cv2
import math


__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 


def load_video(path):
    """ Load all frames from the video at path. """
    cap = cv2.VideoCapture(path)
    frames = []
    f = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
        f += 1

    cap.release()
    return frames

def play_video(vid, 
               title='frame',
               wait=50,
               play_func = cv2.imshow):
    """ Play the video frame list vid. """
    for f in vid:
        play_func(title, f)
        if cv2.waitKey(wait) & 0Xff == ord('q'):
            break
    cv2.destroyWindow(title)

    
def save_video(vid, path, framerate=30.0):
    """ Save the video frame list vid at path.
    Will always use XVID (avi) format."""
    
    dims = vid[0].shape
    print("Saving video to path %s with dimensions %s" % 
          (path, str(dims)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, framerate,
                          (dims[1], dims[0]))
    for f in vid:
        out.write(f)

    print("Saved OK.")
    out.release()
        
def optical_flow(f1, f2):
    """ Compute the optical flow between two frames. Will return
    a matrix of 2-d optical flow vectors for each pixel. """
    
    prev = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    uflow = np.zeros_like(prev)
    flow = cv2.calcOpticalFlowFarneback(prev,
                                        next,
                                        uflow,
                                        0.5, # pyr scale
                                        3,  # levels
                                        25,  # winsize
                                        3, # iterations
                                        7,  # poly_n
                                        1.5,  # ploy_sigma
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    return flow

def optical_flows(fs):
    """ Compute the optical flow for every frame in fs. 
    Return the resulting array. """
    flows = []
    while len(fs) > 1:
        flows.append(optical_flow(fs[0], fs[1]))
        fs = fs[1:]
    return flows

def show_flow(title, f):
    """ Show the optical flow frame f. """
    bgr = flow_to_bgr(f)
    cv2.imshow(title, bgr)

def flow_to_bgr(f):
    """ Converts the optical flow frame f to a viewable
    bgr frame. """
    s = f.shape
    hsv = np.zeros((s[0], s[1], 3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(f[...,0], f[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
    
def play_flows(fs, title='flows', wait=50):
    play_video(fs, title=title, wait=wait, 
               play_func = lambda t, x: show_flow(t, x))
    
def make_every_other_frame_black(fs):
    """ Take a sequence of video frames, fs, and
    insert a black frame between each one."""
    new = []
    for f in fs:
        new.append(f)
        new.append(np.zeros_like(fs[0]))
    return new

def save_frames_to_numpy(fs, path):
    """ Save frames as a single .npy file """
    temp = np.stack(fs, axis = 3)
    np.save(path, temp)
    
def load_numpy_video(path):
    """ Load a video saved as a numpy file, return it 
    as a list of frames. """
    mat = np.load(path)
    fs = []
    for i in range(mat.shape[3]):
        fs.append(mat[:,:,:,i])
    return fs
    
def copy_frames(fs):
    """ return a deep copy of the video frames sequence frames. """
    new_fs = [np.copy(f) for f in fs]
    return new_fs
     
def view(f, title='frame', wait=0):
    """ View a single frame f. """
    while True:
        cv2.imshow(title, f)
        if cv2.waitKey(0) & 0Xff == ord('q'):
            break
    cv2.destroyWindow(title)

def view_frame_by_frame(fs, title='frame', wait=0):
    """ View a video frame by frame """
    if (fs[0].shape[2] == 2):
        fs = [flow_to_bgr(f) for f in fs]
    for f in fs:
        cv2.imshow(title, f)
        key = cv2.waitKey(0) & 0Xff
        if key == ord('q'):
            break
    cv2.destroyWindow(title)

def clip(f, xr, yr):
    """ clip the frame f and return only pixels in the range
    xr, yr """
    if len(xr) == 1:
        f = f[:][xr[0]:][:]
    else:
        f = f[:][xr[0]:xr[1]][:]
    if len(yr) == 1:
        f = f[yr[0]:][:][:]
    else:
        f = f[yr[0]:yr[1]][:][:]
    return f
    
    
def eucdist(c1, c2):
    """ Return euclidean distance between two colors """
    c1 = c1.astype('float')
    c2 = c2.astype('float')
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2
                     + (c1[2]-c2[2])**2)
    
