#! /usr/bin/env python3

""" Simple command line utility for computing optical flow,
saves a video of the flow and the resulting Numpy data. """

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
              help="Input video file path",
              default="./footage/p1080/footballshort.avi")
@click.option("-o", type=click.Path(),
              help="Output path for flows (.npy) and flow video (.avi)",
              default="./flows/farneback/football")

def make_and_save_optical_flow(i, o):
    """ Compute the optical flow of file at inpath,
    output a .npy and a .avi file at outpath. """
    frames = ut.load_video(i)
    print("Calculating flows...")
    flows  = ut.optical_flows(frames)
    print("Complete.")
    print("Saving data... ")
    ut.save_video([ut.flow_to_bgr(f) for f in flows], 
                  o + ".avi")
    ut.save_frames_to_numpy(flows, o + ".npy")
    print("Done.")

    
if __name__ == '__main__':
    make_and_save_optical_flow()


