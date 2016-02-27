#! /usr/bin/env python3

import numpy as np
import cv2
import click

@click.command()
@click.option("--source", type = click.Path(exists=True),
              default="./erlich.mp4")
@click.option("--dest", type = click.Path(),
              default="./out.avi")
@click.option("--start", default=192)
@click.option("--stop", default=0.0)
@click.option("--frames", default=48)

def cut(source, dest, start, stop, frames):
    framecount = 0
    limit = start + frames
    print(start)
    print(limit)
    cap = cv2.VideoCapture(source)
    #fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    ret, first = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dest, fourcc, 23.976024, (first.shape[1], first.shape[0]))
    while(cap.isOpened()):
        ret, frame = cap.read()
        framecount += 1
        print(framecount)
        if ret == True and framecount >= start:
            out.write(frame)
        if framecount >= limit:
            break

    print("Wrote video.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(dest)
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
