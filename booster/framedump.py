#! /usr/bin/env python3

import numpy as np
import booster.utility as ut
import cv2
import click

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
              default="./footage/animesword.mp4")
@click.option("-o", type = click.Path(exists=True),
              default="./footage/animesword.mp4")


def dump(i, o):
    frames = ut.load_video(i)
    c = 0
    for f in frames:
        path = o + "frame%d.ppm" % c
        cv2.imwrite(path, f)
        c += 1
        if c > 30:
            return 


    

if __name__ == '__main__':
    dump()
