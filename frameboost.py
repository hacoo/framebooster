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
import numpy as np
import cv2
import booster.utility as ut
import booster.booster as bst

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
              default="./erlichshort.avi", 
              help="Input video file")
@click.option("-o", type = click.Path(),
              default="./out.avi",
              help="Output file path")


def start(i, o):
    print("Loading video at path:", i)

    frames = ut.load_video(i)            
    #flows  = ut.optical_flows(frames)
    #ut.play_flows(flows)
    flow0 = ut.optical_flow(frames[0], frames[1])
    flow1 = ut.optical_flow(frames[1], frames[2])
    f = frames[0] 
    print(bst.flowdist(flow0[0][0], flow0[0][0]))

    dumbpatch = flow0[0:100, 0:100]
    smartpatch = bst.makepatch(flow0, [50,50], 100)
    while(1):
        cv2.imshow('frame', f)
        cv2.imshow('dumb patch', ut.flow_to_bgr(dumbpatch))
        cv2.imshow('smart patch', ut.flow_to_bgr(smartpatch))
        cv2.moveWindow('smart patch', 100, 100)
        cv2.moveWindow('dumb patch', 300, 100)
        if cv2.waitKey(1000) & 0Xff == ord('q'):
            break
            
    cv2.destroyAllWindows()
    #print(bst.patchdist(smartpatch, dumbpatch))


if __name__ == '__main__':
    start()
