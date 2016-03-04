"""
Functions for dealing with .flo files
"""

import numpy as np
import struct
import cv2


__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 


def load_flo(path):
    """ Load a .flo file, a sequence of [u,v] vectors.
    Return as a 3-d numpy array. """
    
    # PIEH is the sanity byte
    with open(path, 'rb') as f:
        pieh = struct.unpack('f', f.read(4))
        assert(pieh[0] == 202021.25)
        width  = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        print("Loading flows at path %s with dimensions h: %d w: %d" %
              (path, height, width))
        data = np.ndarray((height, width, 2))
        for r in range(height):
            for c in range(width):
                u = struct.unpack('f', f.read(4))[0]
                v = struct.unpack('f', f.read(4))[0]
                data[r][c] = np.array([u,v])
    return data
