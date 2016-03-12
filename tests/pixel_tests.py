""" Tests for the pixel module. """

from nose.tools import *
import booster.pixels as pix
import numpy as np
import booster.utility as ut
import booster.booster as bst
import cv2
import time

image = np.zeros((16, 16, 3), dtype=np.uint8)
image[0, 0, 1] = 255
image[8, 8, 1] = 255
image[15, 15, 1] = 255

small = np.zeros((2,2,3), dtype=np.uint8)


def setup():
    pass

def teardown():
    pass

# def test_pixel_at_coordinates():
#     # check we get the right stuff

#     pixel = pix.pixel_at_coordinates(image, 0.0, 0.0)
#     assert(pixel[1] == 255)

#     pixel = pix.pixel_at_coordinates(image, 7.0, -7.0)
#     assert(pixel[1] == 255)

#     pixel = pix.pixel_at_coordinates(image, 7.0, 8.0)
#     assert(pixel[1] == 0)

#     pixel = pix.pixel_at_coordinates(image, 0.0, 0.0)
#     assert(pixel[1] == 255)

#     pixel = pix.pixel_at_coordinates(image, 1.0, 0.0)
#     assert(pixel[1] == 0)

#     # Check out-of-bounds returns null
#     pixel = pix.pixel_at_coordinates(image, -8.6, 0.0)
#     assert(pixel == None)

#     pixel = pix.pixel_at_coordinates(image, -8.0, 8.6)
#     assert(pixel == None)

#     pixel = pix.pixel_at_coordinates(image, 8.6, 8.6)
#     assert(pixel == None)

# def test_indices_at_coordinates():
#     indices = pix.indices_at_coordinates(small, 0.0, 0.0)
#     assert(indices == (1,1))
#     indices = pix.indices_at_coordinates(small, -1.0, 1.0)
#     assert(indices == (0,0))
#     indices = pix.indices_at_coordinates(small, -0.5, 0.5)
#     assert(indices == (0,0))
    
# def test_splat():

#     splats = pix.splat(small, -1.0, 1.0)

#     assert((0,0) in splats)
#     assert(len(splats) == 1)

#     splats = pix.splat(small, -1.0, 0.9)
#     assert((0,0) in splats)
#     assert((1,0) in splats)
#     assert(len(splats) == 2)

#     splats = pix.splat(small, -0.75, 0.75)
#     assert((0,0) in splats)
#     assert((1,0) in splats)
#     assert((0,1) in splats)
#     assert(len(splats) == 3)

#     splats = pix.splat(small, -0.5, 0.5)
#     assert((0,0) in splats)
#     assert((1,0) in splats)
#     assert((0,1) in splats)
#     assert(len(splats) == 3)

# def test_send_motions():
#     bar = np.zeros((100, 100, 2))
#     bar[40:60, :] = [25.0, 0.0]
#     interleaved = bst.interleave_pointwise_motion([bar])
#     #ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in interleaved])
    
def test_interleave_splat_motions():
    # bar0 = np.zeros((100, 100, 3), dtype=np.uint8)
    # bar1 = np.zeros((100, 100, 3), dtype=np.uint8)
    # bar0[30:40, :] = [100, 100, 0]
    # bar1[50:60, :] = [100, 100, 0]
    # bars = [bar0, bar1]
    # flows = ut.optical_flows(bars)
    # ut.view_frame_by_frame(bars)
    # ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in flows])
    
    bar0 = np.zeros((100,100,2))
    bar0[30:40, :] = [0.0, -10.0]
    bar1 = np.zeros((100,100,2))
    bar1[60:70, :] = [0.0, -10.0]

    bars = [bar0, bar1]
    interleaved = bst.interleave_splat_motion(bars)
    ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in interleaved])
    
def test_image_to_cartesian():
    img = np.zeros((50, 100, 2))
    assert(pix.image_to_cartesian(img, 0, 0) == (-0.5, 0.5))
    bl = pix.image_to_cartesian(img, 49, 99)
    # run epsilon close tests
    assert (bl[0] - 0.5 < 0.1)
    assert (bl[1] - (-0.5) < 0.1)
    
def test_cartesian_to_image():
    img = np.zeros((50, 100, 2))
    bl = pix.image_to_cartesian(img, 49, 99)
    assert (bl[0] - 49 < 0.01)
    assert (bl[1] - 99 < 0.01)
    
def test_cartesian_to_index():
    img = np.zeros((50, 100, 2))
    bl = pix.image_to_cartesian(img, 99, 49)
    bl = pix.cartesian_to_index(img, bl[0], bl[1])
    assert (bl[0] == 99)
    assert (bl[1] == 49)
    bl = pix.image_to_cartesian(img, 0, 0)
    bl = pix.cartesian_to_index(img, bl[0], bl[1])
    assert (bl[0] == 0)
    assert (bl[1] == 0)
    bl = pix.image_to_cartesian(img, 50, 25)
    bl = pix.cartesian_to_index(img, bl[0], bl[1])
    assert (bl[0] == 50)
    assert (bl[1] == 25)

