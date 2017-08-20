# framebooster

Framebooster is a optical-flow based video frame interpolator. Given
a video, it will interpolate intermediate frames -- resulting in a video
with the frame rate doubled. It is written in Python with OpenCV.

If you're just interested in results, take a look in the example_outputs folder. The results are fairly good, although some edge artifacts remain. Note that output videos will appear to be in slow motion, as they have twice as many frames, but the same playback rate.

I use the Large-Displacement Optical Flow (LDOF) algorithm to compute optical flows; in my testing, this resultid in considerably better quality than other methods (e.g., Lucas-Kanade). You can find the LDOF paper here: https://lmb.informatik.uni-freiburg.de/people/brox/pub/brox_tpami10_ldof.pdf

Setup:

This program requires Python 3 and OpenCV. See http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
for details on installing OpenCV.

Framebooster requires the following Python additional packages: numpy, tqdm, click, pick, glob. These packages should all be available via Pip3, e.g.:

> sudo pip3 install tqdm

Framebooster uses the Large Displacement Optical Flow (LDOF) to calculate
optical flows. LDOF must be installed if you want to calculate optical
flows using this program.

LDOF is freely available precompiled or from source, and can be found here:

http://lmb.informatik.uni-freiburg.de/resources/software.php

Once you have the the LDOF binary executable, simply point framebooster.py
to it using the --ldof-path option.

Running framebooster:

example: 

> ./framebooster.py -i "videos/example.mp4" --ldof-path="denseflow/ldof"

This would generate output video in the local directory, out.avi.

For help, run:

> ./framebooster.py --help

Framebooster may be run with  the following options:

  -i : The input video file path.

  -o : The output video file path. By default, will write to ./out.avi

  --calc-flows / --no-calc-flows : Turns optical flow calculation on
  or off. If turned off, flows will not be recomputed, the program
  will look for precomputed flows in the forward/ and backward/ 
  directories.

  --ldof-path : Path to the ldof executable (default: ./denseflow/ldof)

  --clean/--no-clean : Deletes frames and flows from the backward,	
  forward, and frames directories before running. By default,
  this flag is set off.

  --nframes : If set to 0 (default), will process the whole input	
  video. Otherwise, only the first n frames are processed.
  

