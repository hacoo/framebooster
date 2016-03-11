""" 
Main framerate booster module.
"""

import numpy as np
import cv2
import booster.utility as ut
import booster.pixels as pix
import booster.color_transfer as ct
from tqdm import tqdm

__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 


def interleave_splat_motion(fs, t=0.5):
    """ Take a sequences of flow frames, fs, and image
    frames, ims, return a sequence of flow frames where every other
    frame is the splat-interpolated flow frame. """
    pbar = tqdm(total=len(fs))
    interleaved = []
    while fs != []:
        f = fs[0]
        interleaved.append(f)
        interleaved.append(pix.splat_motions(f, t))
        fs = fs[1:]
        pbar.update(1)
    return interleaved

def interleave_pointwise_motion(fs, t=0.5):
    """ Simpler interleave function, simply sends each
    motion vector to the nearest pixel in the target image. """
    interleaved = []
    while fs != []:
        f = fs[0]
        interleaved.append(f)
        interleaved.append(pix.send_motions(f, t))
        fs = fs[1:]
    return interleaved

def interpolate_frames_precomputed_flows(fs, flows, t=0.5):
    """ Take a sequence of frames, fs, with flows, flows, and returns
    a new sequence with interleaved interpolated frames.
    Returns a tuple with the interleaved frames and flows."""
    pbar = tqdm(total=len(fs))
    interleaved = []
    iflows = []
    while len(fs) > 1:
        f0 = fs[0]
        f1 = fs[1]
        fl0 = flows[0]
        flow = pix.splat_motions(fl0, t)
        interpolated = ct.transfer_colors(f0, f1, flow, t)
        interleaved.append(f0)
        interleaved.append(interpolated)
        iflows.append(fl0)
        iflows.append(flow)
        fs = fs[1:]
        flows = flows[1:]
        pbar.update(1)
    return (interleaved, iflows)

def interpolate_frames(frames, flows, fwarped, t=0.5):
    """ Take a sequence of frames, fs, with flows, flows, and returns
    a new sequence with interleaved interpolated frames."""

    pbar = tqdm(total=len(frames)-1)
    interleaved = []
    while len(frames) > 1:
        f0 = frames[0]
        f1 = frames[1]
        flow = fwarped[0]
        interpolated = ct.transfer_colors(f0, f1, flow, t)
        interleaved.append(f0)
        interleaved.append(interpolated)
        frames = frames[1:]
        fwarped = fwarped[1:]
        flows = flows[1:]
        pbar.update(1)

    interleaved.append(frames[0])
    return interleaved


def interpolate_frames_weighted(fs, t=0.5):
    """ 
    Take a sequence of frames, fs, with flows, flows, and returns
    a new sequence with interleaved interpolated frames.
    Returns a tuple with the interleaved frames and flows.
    Uses weighted color transfer a la Shiratori et al.
    """

    pbar = tqdm(total=len(fs)-1)
    interleaved = []
    iflows = []
    while len(fs) > 1:
        f0 = fs[0]
        f1 = fs[1]
        fl0 = ut.optical_flow(f0, f1)
        flow = pix.splat_motions(fl0, t)

        interpolated = ct.color_transfer_all_weighted(f0, fl0, flow, t)
        interleaved.append(f0)
        interleaved.append(interpolated)
        iflows.append(fl0)
        iflows.append(flow)
        fs = fs[1:]
        pbar.update(1)

    interleaved.append(fs[0])
    return (interleaved, iflows)


def interpolate_frames_bidi(frames, forwards, backwards, t=0.5):
    """ Interpolate frames, using bidirectional flows and 
    photoconsistency. """
    pbar = tqdm(total=len(frames)-1)
    interleaved = []
    iflows = []
    while len(frames) > 1:
        flow = pix.splat_motions_bidi(forwards[0], backwards[0],
                                      frames[0], frames[1], t)
        flow = cv2.GaussianBlur(flow, (7,7), 8)
        interpolated = ct.transfer_colors(frames[0], frames[1], flow, t)
        interleaved.append(frames[0])
        interleaved.append(interpolated)
        iflows.append(forwards[0])
        iflows.append(flow)
        frames = frames[1:]
        forwards = forwards[1:]
        backwards = backwards[1:]
        pbar.update(1)
        interleaved.append(frames[0])
        return (interleaved, iflows)


def interpolate_frames_occlusions(frames, forwards, backwards, t=0.5):
    """ Interpolate frames, using bidirectional flows and 
    photoconsistency. """
    pbar = tqdm(total=len(frames)-1)
    interleaved = []
    iflows = []
    while len(frames) > 1:
        flow = pix.splat_motions_bidi(forwards[0], backwards[0],
                                      frames[0], frames[1], t)
        flow = cv2.GaussianBlur(flow, (11, 11), 10)
        interpolated = ct.color_transfer_occlusions(frames[0],
                                                    frames[1],
                                                    forwards[0],
                                                    backwards[0],
                                                    flow,
                                                    t)
        interleaved.append(frames[0])
        interleaved.append(interpolated)
        #iflows.append(forwards[0])
        iflows.append(flow)
        frames = frames[1:]
        forwards = forwards[1:]
        backwards = backwards[1:]
        pbar.update(1)
        ut.save_frames_to_numpy(iflows, "./flows/interleaved/flows.npy")
    return (interleaved, iflows)



def fwarp_flows(flows, t=0.5):
    """ Make a sequence of forward-warped flows.
    Not interpolated. """

    iflows = []
    while flows != []:
        iflows.append(pix.splat_motions(flows[0], t))
        flows = flows[1:]
    return iflows


        



