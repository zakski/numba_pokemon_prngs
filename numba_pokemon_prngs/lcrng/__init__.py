"""Linear Congruential Pseudo Random Number Generators"""
import numba
from .lcrng32 import (
    LCRNG32,
    LCRNG32RandomDistribution,
    lcrng32_init,
)

# pylint: disable=abstract-method


@numba.experimental.jitclass()
@lcrng32_init(
    add=0x6073, mult=0x41C64E6D, distribution=LCRNG32RandomDistribution.MODULO
)
class PokeRNGMod(LCRNG32):
    """Standard Pokemon LCRNG with modulo random distribution"""


@numba.experimental.jitclass()
@lcrng32_init(
    add=0x6073,
    mult=0x41C64E6D,
    distribution=LCRNG32RandomDistribution.RECIPROCAL_DIVISION,
)
class PokeRNGDiv(LCRNG32):
    """Standard Pokemon LCRNG with reciprocal division random distribution"""


# pylint: enable=abstract-method
