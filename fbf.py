#! /usr/bin/env python3

""" view a video frame by frame. """
import numpy as np
import cv2
import click
import booster.utility as ut

__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 

@click.command()
@click.option("-i", type=click.Path(exists=True),
              help="Input video path",
              default="./flows/interleaved/football.npy")
@click.option("-n", type=bool,
              help="Video is a numpy file",
              default=False)



def fbf(i, n):
    if n:
        frames = ut.load_numpy_video(i)
    else:
        frames = ut.load_video(i)

    if frames[0].shape[2] == 2:
        frames = [ut.flow_to_bgr(f) for f in frames]
    ut.view_frame_by_frame(frames)

if __name__ == '__main__':
    fbf()
