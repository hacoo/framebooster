"""
Functions for transfering color to the interpolated image. 
"""
import numpy as np
import cv2
from tqdm import tqdm
import booster.pixels as pix
import booster.utility as ut
import booster.geometry as geo
import math


__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 



def transfer_color(f0, f1, dest, flow, row, col, t=0.5):
    """ Find and set the transfer color for pixel [row][cow] """
    # for now, just defauly to using the value at f1:
    h = dest.shape[0]
    w = dest.shape[1]
    u = flow[row][col]
    ux = u[0]/w
    uy = -u[1]/h
    (x, y) = pix.image_to_cartesian(flow, col, row)
    xp0 = x - t*ux
    yp0 = y - t*uy
    xp1 = x + t*ux
    yp1 = y + t*uy
    (xi0, yi0) = pix.cartesian_to_index(flow, xp0, yp0)
    (xi1, yi1) = pix.cartesian_to_index(flow, xp1, yp1)
    if (pix.check_indices(flow, xi0, yi0) and
        pix.check_indices(flow, xi1, yi1)):
        #dest[row][col] = (f0[yi0][xi0])/2 + (f1[yi1][xi1])/2
        dest[row][col] = f0[yi0][xi0]

    

def transfer_colors(f0, f1, flow, t=0.5):
    """ Interpolate colors using frames f1 and f0
    onto flows, and return the interpolated frame."""
    h = f0.shape[0]
    w = f0.shape[1]
    dest = np.zeros_like(f1)
    for row in tqdm(range(h), nested=True, desc="ct"):
        for col in range(w):
            transfer_color(f0, f1, dest, flow, row, col, t)
    return dest

def color_transfer_all_weighted(f0, flow0, flowt, t=0.5):
    h = f0.shape[0]
    w = f0.shape[1]
    dest = np.zeros_like(f0, dtype = 'float64')
    ws   = np.zeros_like(dest)
    for row in tqdm(range(h), nested=True, desc="ct"):
        for col in range(w):
            color_transfer_weighted(f0, flow0, flowt, dest,
                                    row, col, ws, t)
    # normalize by weights
    for row in range(h):
        for col in range(w):
            dest[row][col] /= ws[row][col]

            
    # convert back to uint8
    return dest.astype('uint8')



def color_transfer_weighted(frame, flow0, flowt, dest,
                            row, col, ws, t=0.5):
    """ Transfer coloring using the Shiratori et al.
    weighted blending technique. In this case [row][col]
    refers to the f0 pixel, not the output pixel."""
    h = frame.shape[0]
    w = frame.shape[1]
     
    # Get the flow for previous timestwp
    q_flow = flow0[row][col]
    # Get the previous frame cartesian coords
    (qx, qy) = pix.image_to_cartesian(frame, col, row)
    
    ux = q_flow[0]/w
    uy = -q_flow[1]/h

    # These are the target coordinates in the interpolated
    # image. We need to find all frames within 0.5 pixels
    # of this spot
    px = qx + t*ux
    py = qy + t*uy
    
    # These are the indices of all the target pixels. We will
    # add to each one...
    ipixels = pix.splat(dest, px, py)
    if ipixels:
        for p in ipixels:
            if (pix.check_indices(dest, p[0], p[1])):
                p_flow = flowt[p[1]][p[0]]
                r = pix.flowdist(p_flow, q_flow)
                add_weighted_color(frame[row][col], p, dest,
                                   px, py, ws, r, t=0.5)
    
def add_weighted_color(q, p, dest, px, py, ws, r, t=0.5):
    """ Transfer weighted color from previous frame pixel
    q onto destination frame pixel at p, given that
    the motion vector's true destination was (px, py) 
    
    r is the flow distance function for p and q.
    """
    # The center coordinates
    row = p[1]
    col = p[0]
    # Maximum distance is 0.5
    distance = geo.vecnorm(col-px, row-py)
    wdist    = 1-distance
    weight   = wdist * r
    dest[row][col] += weight * q
    ws[row][col] += weight
    


def masked_transfer_color(f0, f1, dest, flow, imask, row, col, t=0.5):
    """ Find and set the transfer color for pixel [row][cow] """
    # for now, just defauly to using the value at f1:
    h = dest.shape[0]
    w = dest.shape[1]
    u = flow[row][col]
    ux = u[0]/w
    uy = -u[1]/h
    (x, y) = pix.image_to_cartesian(flow, col, row)
    xp0 = x - t*ux
    yp0 = y - t*uy
    xp1 = x + t*ux
    yp1 = y + t*uy
    (xi0, yi0) = pix.cartesian_to_index(flow, xp0, yp0)
    (xi1, yi1) = pix.cartesian_to_index(flow, xp1, yp1)
    if (pix.check_indices(flow, xi0, yi0) and
        pix.check_indices(flow, xi1, yi1)):
        f0vis = imask[row][col][0]
        f1vis = imask[row][col][1]
        if f0vis >= 1 and f1vis >= 1:
            dest[row][col] = (f0[yi0][xi0])/2 + (f1[yi1][xi1])/2
        elif f1vis >= 1:
            dest[row][col] = f1[yi1][xi1]
        else:
            dest[row][col] = f0[yi0][xi0]
            
    
def color_transfer_occlusions(f0, f1, forward, backward, interp,
                              t=0.5):
    """ Transfer colors to the intermediate frame between f0 and
    f1, with interpolated flow interp. Returns the interpolated
    frame. """
    h = interp.shape[0]
    w = interp.shape[1]
    new_frame = np.copy(f0)
    imask = find_occlusions(forward, backward, interp, t)
    for row in tqdm(range(h), nested=True, desc="color transfer"):
        for col in range(w):
            masked_transfer_color(f0, f1, new_frame, interp, imask,
                                 row, col, t)
    return new_frame
    
def find_occlusions(forward, backward, interp,
                    t=0.5):
    """ Generate the occlusion mask for the interpolated frame. """
    h = interp.shape[0]
    w = interp.shape[1]
    imask = np.zeros((h, w, 2), dtype='uint8')
    for row in tqdm(range(h), nested=True, desc="generating occlusion mask"):
        for col in range(w):
            find_occlusion(forward, backward, interp, imask,
                           row, col, t)
    return imask

def find_occlusion(forward, backward, interp, imask, row, col, t=0.5):
    h = interp.shape[0]
    w = interp.shape[1]
    im = interp[row][col]
    ux = im[0]/w
    uy = -im[1]/h
    (x, y) = pix.image_to_cartesian(interp, col, row)
    xp0 = x - t*ux
    yp0 = y - t*uy
    xp1 = x + t*ux
    yp1 = y + t*uy
    # Get each motion vector
    (xi0, yi0) = pix.cartesian_to_index(forward, xp0, yp0)
    (xi1, yi1) = pix.cartesian_to_index(backward, xp1, yp1)
    if (pix.check_indices(forward, xi0, yi0) and
        pix.check_indices(backward, xi1, yi1)):

        fvec = forward[yi0][xi0]
        bvec = -1*backward[yi1][xi1]
        
        d0 = pix.flowdist(fvec, im)
        d1 = pix.flowdist(bvec, im)
        diff = abs(d0-d1)

        if d0 <= d1:
            imask[row][col][0] = 1
            if diff < 0.1:
                imask[row][col][1] = 1
        else:
            imask[row][col][1] = 1
            if diff < 0.1:
                imask[row][col][0] = 1
    else:
        # Default to using first frame
        imask[row][col][0] = 1

    

def eucdist(u, v):
    """ Return the euclidean distance between vectors u and v """
    return math.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)
    
