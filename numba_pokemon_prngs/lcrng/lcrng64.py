"""64bit Linear Congruential Pseudo Random Number Generator"""

from __future__ import annotations
from typing import Type, Callable
import enum
import numba
import numpy as np


class LCRNG64RandomDistribution(enum.IntEnum):
    """
    Various LCRNG64 random distribution types

    MULTIPLICATION_SHIFT -> (next_u32() * maximum) >> 32
    """

    NONE = 0
    MULTIPLICATION_SHIFT = 1


class LCRNG64:
    """64-bit LCRNG parent class"""

    seed: numba.uint64

    def __init__(self, seed: np.uint64) -> None:
        self.seed: np.uint64 = seed

    def next(self) -> np.uint64:
        """
        Generate and return the next full 64-bit random uint

        () -> seed = seed * mult + add
        """
        raise NotImplementedError()

    def jump(self, adv: np.uint64) -> np.uint64:
        """Jump ahead the LCRNG sequence by adv"""
        raise NotImplementedError()

    def advance(self, adv: np.uint64) -> np.uint64:
        """Advance the LCRNG sequence by adv"""
        for _ in range(adv):
            self.next()
        return np.uint64(self.seed)

    def next_u32(self) -> np.uint32:
        """Generate and return the next 32-bit random uint"""
        return np.uint32(np.uint64(self.next()) >> np.uint64(32))

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        """
        Generate and return the next [0, maximum) random uint

        () -> distribution(self.next_u32(), maximum)
        """
        raise NotImplementedError()


def lcrng64_init(
    *,
    add: np.uint64,
    mult: np.uint64,
    distribution: LCRNG64RandomDistribution = LCRNG64RandomDistribution.NONE,
    reverse: bool = False,
) -> Callable[[Type[LCRNG64]], Type[LCRNG64]]:
    """Initialize a LCRNG64 class with constants and random distribution"""
    if reverse:
        mult = np.uint64(pow(mult, -1, 0x10000000000000000))
        add = np.uint64(-np.uint64(add) * mult)
    else:
        mult = np.uint64(mult)
        add = np.uint64(add)

    jump_table = [(np.uint64(add), np.uint64(mult))]
    for i in range(63):
        jump_table.append(
            (
                np.uint64(jump_table[i][0] * (jump_table[i][1] + 1)),
                np.uint64(jump_table[i][1] * jump_table[i][1]),
            )
        )
    jump_table = tuple(jump_table)

    def wrap(lcrng_class: Type[LCRNG64]) -> Type[LCRNG64]:
        def next_(self: LCRNG64) -> np.uint32:
            self.seed = np.uint64(
                np.uint64(self.seed) * np.uint64(mult) + np.uint64(add)
            )
            return np.uint64(self.seed)

        def jump(self: LCRNG64, adv: np.uint64):
            i = 0
            while adv:
                if adv & 1:
                    add, mult = jump_table[i]
                    self.seed = np.uint64(
                        np.uint64(self.seed) * np.uint64(mult) + np.uint64(add)
                    )
                adv >>= 1
                i += 1
            return np.uint64(self.seed)

        if distribution == LCRNG64RandomDistribution.MULTIPLICATION_SHIFT:

            def next_rand(self: LCRNG64, maximum: np.uint32) -> np.uint32:
                return np.uint32(
                    (np.uint64(self.next_u32()) * np.uint64(maximum)) >> np.uint64(32)
                )

        else:

            def next_rand(self: LCRNG64, maximum: np.uint32) -> np.uint32:
                raise NotImplementedError()

        lcrng_class.next = next_
        lcrng_class.jump = jump
        lcrng_class.next_rand = next_rand

        lcrng_class = numba.experimental.jitclass(lcrng_class)
        return lcrng_class

    return wrap
