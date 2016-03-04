#! /usr/bin/env python3

""" 
Boosts framerate of 24fps videos to 48fps. Implements Motion Field Transfer Completion
as described in the following paper:

Video Completion by Motion Field Transfer
Shiratori, Matsushita, Kang, Tang

IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2006)

Online: http://www.cs.cmu.edu/~./siratori/MotionFieldTransfer/index.html
"""

import click
import cProfile
import numpy as np
import cv2
import pickle
import booster.utility as ut
import booster.booster as bst
import booster.color_transfer as ct
import booster.flo as flo

__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 


@click.command()
@click.option("-i", type = click.Path(exists=True),
              default="./footage/p1080/footballshort.avi", 
              help="Input video file")
@click.option("-o", type = click.Path(),
              default="./footage/p1080/out.avi",
              help="Output file path")


def start(i, o):
    print("Loading video at path:", i)
    # frames = ut.load_video("./footage/p1080/footballshort.avi")
    # frames = frames[4:8]
    # frames = [f[400:, 1200:, : ] for f in frames]
    # ut.view_frame_by_frame(frames)
    # cv2.imwrite("./frame4.ppm", frames[0])
    # cv2.imwrite("./frame5.ppm", frames[1])
    # cv2.imwrite("./frame6.ppm", frames[2])
    # cv2.imwrite("./frame7.ppm", frames[3])

    # ut.view_frame_by_frame(frames)
    # forward = [flo.load_flo("./denseflow/forwardtall.flo")]
    # ut.view_frame_by_frame(forward)
    # forward = [cv2.GaussianBlur(f, (7,7), 8) for f in forward]
    # ut.view_frame_by_frame(forward)
    # backward = [flo.load_flo("./denseflow/backwardtall.flo")]
    # backward= [cv2.GaussianBlur(b, (7,7), 8) for b in backward]
    # frames = [cv2.imread("./frame0tall.ppm"),
    #           cv2.imread("./frame1tall.ppm")]

    forward = [flo.load_flo("./denseflow/frame4forward.flo"),
               flo.load_flo("./denseflow/frame5forward.flo"),
               flo.load_flo("./denseflow/frame6forward.flo")]
    ut.view_frame_by_frame(forward)
    forward = [cv2.GaussianBlur(f, (7,7), 8) for f in forward]
    backward = [flo.load_flo("./denseflow/frame4backward.flo"),
               flo.load_flo("./denseflow/frame5backward.flo"),
               flo.load_flo("./denseflow/frame6backward.flo")]

    backward= [cv2.GaussianBlur(b, (7,7), 8) for b in backward]
    ut.view_frame_by_frame(backward)
    frames = [cv2.imread("./frame4.ppm"),
              cv2.imread("./frame5.ppm"),
              cv2.imread("./frame6.ppm"),
              cv2.imread("./frame7.ppm")]
    ut.view_frame_by_frame(frames)

    (interleaved, iflows) = bst.interpolate_frames_occlusions(frames, 
                                                              forward,
                                                              backward)
    ut.view_frame_by_frame(interleaved)
    ut.view_frame_by_frame(iflows)
    ut.save_video(interleaved, "./footage/output/ptc.avi")
    ut.save_frames_to_numpy(iflows, "./flows/interleaved/ptc.npy")
    return

 #   flows = [cv2.GaussianBlur(f, (7, 7), 2) for f in flows]
 #    fwarps = [cv2.GaussianBlur(f, (7, 7), 2) for f in fwarps]
    
    #flows = ut.load_numpy_video("./flows/interleaved/football_full.npy")
    
    #ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in flows])



    ut.save_frames_to_numpy(iframes, "./footage/output/football_2f.npy")
    ut.save_video(iframes, "./footage/output/football_2f.avi")
    ut.save_frames_to_numpy(iflows, 
                            "./flows/interleaved/football_2f.avi")
    ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in iflows])
    ut.view_frame_by_frame(iframes)
    #frs = frames[0:2]
    #frs = [f[500:800, 1400:, :] for f in frs]
    #ut.view_frame_by_frame(frs)

    #flows = ut.load_numpy_video("./flows/farneback/footballshort.npy")
    #fs = flows[0:1]
    #fs = [f[500:800, 1400:, :] for f in fs]
    #interleaved = bst.interleave_splat_motion(fs)
    #flows = ut.load_numpy_video("./flows/interleaved/football.npy")
    
    #ut.save_frames_to_numpy(interleaved,
    #                       "./flows/interleaved/football.npy")
    
     

    #ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in interleaved])
    #flows = ut.load_numpy_video("./flows/interleaved/football.npy")
    #flows = flows[0:2]
    
    #ut.view_frame_by_frame(frs)
    #interleaved = ct.transfer_colors(frs[0], frs[1], flows[1])
    #ut.view_frame_by_frame([frs[0], interleaved, frs[1]])
    #ut.view_frame_by_frame([ut.flow_to_bgr(f) for f in flows])


    #ut.save_video(interleaved, "./footage/output/football.avi")
    #ut.view_frame_by_frame(interleaved)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    start()
