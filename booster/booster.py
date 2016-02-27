""" 
Main framerate booster module.
"""

import numpy as np
import cv2
import booster.utility as u

__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 





def flowdist(m0, m1):
    """ Compute the distance between two optical flow vectors,
    m0 and m1. """

    # Convert to homogeneous coordinates, assuming timestep 
    # is just 1:

    assert(m0.shape == (2,))
    assert(m1.shape == (2,))
    m0h = np.array([m0[0], m0[1], 1])
    m1h = np.array([m1[0], m1[1], 1])
    top = m0h.dot(m1h)
    m0norm = np.linalg.norm(m0)
    m1norm = np.linalg.norm(m1)    
    return 1 - (top / (m0norm * m1norm))


def patchdist(p0, p1):
    """ Compute the distance between two patches. Patches
    are 2d numpy arrays of flow vectors. """
    assert(p0.shape == p1.shape)
    D = p0.size
    sum = 0
    for i in range(p0.shape[0]):
        for j in range(p1.shape[1]):
            sum += flowdist(p0[i, j], p1[i, j])
    return sum / D

def makepatch(image, c, w):
    """ make a patch from the image, centered at c and with
    width w """
    rad = w // 2
    cx = c[0]
    cy = c[1]
    height = image.shape[0]
    width  = image.shape[1]
    patch  = np.zeros((w,w,image.shape[2]), dtype=image.dtype)
    
    for x in range(w):
            px = cx + x - rad
            for y in range(w):
                py = cy + y - rad
                if (0 <= py <= height) and (0 <= px <= width):
                    patch[y, x] = image[py, px]
    return patch
            
    
    
        
        
    
