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
import glob, os
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
              default="./footage/p480/pizzaratshort.avi",
              help="Input video path")
@click.option("-o", type = click.Path(),
              default="./output/out",
              help="Output video path")
@click.option("--calc-flows/--no-calc-flows", 
              default=True,
              help="Recalculate optical flows")
@click.option("--ldof-path", type = click.Path(exists=True),
              default="./denseflow/ldof",
              help="ldof optical flow calculator executable path")
@click.option("--clean/--no-clean", default = False, 
              help = "Clean directories before running") 
@click.option("--nframes", default=0, type = click.IntRange(0),
              help = "Number of frames to process. 0 will process whole video") 






def start(i, o, calc_flows, ldof_path, clean, nframes):
    if clean:
        clean_directories()

    frames = ut.load_video(i)
    if nframes != 0:
        frames = frames[0:nframes]
    ut.save_video_frames(frames, "./frames")
    if calc_flows:
        ut.calc_flows_forward_brox(frames, "./frames", "./forward", 
                                   ldof_path)
        ut.calc_flows_backward_brox(frames, "./frames", "./backward",
                                    ldof_path)

    forward = ut.load_forward_flows(frames, "./forward")
    ut.view_frame_by_frame(forward)
    
    backward = ut.load_backward_flows(frames, "./backward")
    ut.view_frame_by_frame(backward)


    ut.view_frame_by_frame(frames)
    # forward = [flo.load_flo("./denseflow/frame4forward.flo"),
    #            flo.load_flo("./denseflow/frame5forward.flo"),
    #            flo.load_flo("./denseflow/frame6forward.flo")]
    # ut.view_frame_by_frame(forward)
    # forward = [cv2.GaussianBlur(f, (7,7), 8) for f in forward]
    # backward = [flo.load_flo("./denseflow/frame4backward.flo"),
    #            flo.load_flo("./denseflow/frame5backward.flo"),
    #            flo.load_flo("./denseflow/frame6backward.flo")]

    # backward= [cv2.GaussianBlur(b, (7,7), 8) for b in backward]
    # ut.view_frame_by_frame(backward)
    # frames = [cv2.imread("./frame4.ppm"),
    #           cv2.imread("./frame5.ppm"),
    #           cv2.imread("./frame6.ppm"),
    #           cv2.imread("./frame7.ppm")]
    # ut.view_frame_by_frame(frames)
    
    (interleaved, iflows) = bst.interpolate_frames_occlusions(frames, 
                                                              forward,
                                                              backward)
    ut.view_frame_by_frame(interleaved)
    ut.view_frame_by_frame(iflows)
    ut.save_video(interleaved, o+".avi")
    ut.save_frames_to_numpy(iflows, o+".npy")
    return

def clean_directories():
    """ Clean out working directories to prepare for a run """
    for dir in ["frames", "backward", "forward"]:
        files = glob.glob(dir+"/*")
        for f in files:
            os.remove(f) 

if __name__ == '__main__':
    start()
