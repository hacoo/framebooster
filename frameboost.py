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
              default="./in.avi",
              help="Input video path")
@click.option("-o", type = click.Path(),
              default="./out",
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
    set_up_directories()
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
    backward = ut.load_backward_flows(frames, "./backward")

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
        files = glob.glob(dir+"/*.ppm")
        for f in files:
            os.remove(f) 
        files = glob.glob(dir+"/*.flo")
        for f in files:
            os.remove(f) 


def set_up_directories():
    """ make sure needed directories are present. """
    for dir in ["./frames", "./forward", "./backward"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
            

if __name__ == '__main__':
    start()
