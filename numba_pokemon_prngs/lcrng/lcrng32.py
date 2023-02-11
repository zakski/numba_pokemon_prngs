"""32-bit Linear Congruential Pseudo Random Number Generator"""

from __future__ import annotations
from typing import Type, Callable
import enum
import numba
import numpy as np


class LCRNG32RandomDistribution(enum.IntEnum):
    """
    Various LCRNG32 random distribution types

    MODULO -> next_u16() % maximum

    RECIPROCAL_DIVISION -> next_u16() // ((0xFFFF // maximum) + 1)
    """

    NONE = 0
    MODULO = 1
    RECIPROCAL_DIVISION = 2


class LCRNG32:
    """32-bit LCRNG parent class"""

    seed: numba.uint32

    def __init__(self, seed: np.uint32) -> None:
        self.seed: np.uint32 = seed

    def next(self) -> np.uint32:
        """
        Generate and return the next full 32-bit random uint

        () -> seed = seed * mult + add
        """
        raise NotImplementedError()

    def next_u16(self) -> np.uint16:
        """Generate and return the next 16-bit random uint"""
        return self.next() >> 16

    def next_rand(self, maximum: np.uint16) -> np.uint16:
        """
        Generate and return the next [0, maximum) random uint

        () -> distribution(self.next_u16(), maximum)
        """
        raise NotImplementedError()


def lcrng32_init(
    *,
    add: np.uint32,
    mult: np.uint32,
    distribution: LCRNG32RandomDistribution = LCRNG32RandomDistribution.NONE,
) -> Callable[[Type[LCRNG32]], Type[LCRNG32]]:
    """Initialize a LCRNG32 class with constants and random distribution"""

    def wrap(lcrng_class: Type[LCRNG32]) -> Type[LCRNG32]:
        def next_(self: LCRNG32) -> np.uint32:
            self.seed = self.seed * mult + add
            return self.seed

        if distribution == LCRNG32RandomDistribution.MODULO:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                return self.next_u16() % maximum

        elif distribution == LCRNG32RandomDistribution.RECIPROCAL_DIVISION:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                return self.next_u16() // ((0xFFFF // maximum) + 1)

        else:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                raise NotImplementedError()

        lcrng_class.next = next_
        lcrng_class.next_rand = next_rand
        return lcrng_class

    return wrap
