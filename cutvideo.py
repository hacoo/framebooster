#! /usr/bin/env python3

import numpy as np
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
              default="./football1080.mp4")
@click.option("-o", type = click.Path(),
              default="./out.avi")
@click.option("--start", default=33.0)
@click.option("--stop", default=34.0)
@click.option("--framerate", default = 29.97)

def cut(i, o, start, stop, framerate):
    framecount = 0
    start = start*framerate
    stop  = stop*framerate
    limit = stop - start
    print(start)
    print(stop)
    print(limit)
    cap = cv2.VideoCapture(i)
    #fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    ret, first = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(o, fourcc, framerate,
                          (first.shape[1], first.shape[0]))
    while(cap.isOpened()):
        ret, frame = cap.read()
        framecount += 1
        if ret == True and framecount >= start:
            out.write(frame)
            if framecount >= stop:
                break

    print("Wrote video.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(o)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    cut()
