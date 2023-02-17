"""Package for numba @jitclass implementations
of the Pseudo-Random Number Generators used in the Pokemon series"""
import numpy as np

np.seterr(over="ignore")
del np

from .rng_list import build_rnglist, RNGList
from .sha1 import SHA1
