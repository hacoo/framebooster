
"""
Functions for basic geometry stuff.
"""

import numpy as np
import math

__author__     = "Henry Cooney"
__credits__    = ["Henry Cooney", "Feng Liu"]
__license__    = "MIT"
__version__    = "0.1"
__maintainer__ = "Henry Cooney"
__email__      = "hacoo36@gmail.com"
__status__     = "Prototype"
__repo__       = "https://github.com/hacoo/framebooster.git" 


def vecnorm(dx, dy):
    """ Return the length of vector with components dx, dy """
    return math.sqrt(dx*dx + dy*dy)
    
