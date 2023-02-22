"""SIMD-oriented Fast Mersenne Twister 19937 Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..compilation import optional_jitclass, array_type


# TODO: jump tables
# TODO: staticmethod const functions
@optional_jitclass
class SIMDFastMersenneTwister:
    """SIMD-oriented Fast Mersenne Twister Pseudo Random Number Generator"""

    state: array_type(np.uint32)  # contiguous array
    index: np.uint16

    def __init__(self, seed: np.uint32) -> None:
        seed = np.uint32(seed)
        self.state = np.empty(624, dtype=np.uint32)
        self.index = np.uint16(624)  # ensures shuffle after initialization
        self.re_init(seed)

    def re_init(self, seed: np.uint32) -> None:
        """Reinitialize without creating a new object"""
        seed = np.uint32(seed)
        self.state[0] = seed
        self.index = np.uint16(624)  # ensures shuffle after initialization
        inner = np.uint32(seed & np.uint32(1))

        for i in range(1, 624):
            seed = np.uint32(
                np.uint32(0x6C078965) * (seed ^ (seed >> np.uint32(30))) + np.uint32(i)
            )
            self.state[i] = seed

        inner ^= self.state[3] & np.uint32(0x13C9E684)
        inner ^= inner >> 16
        inner ^= inner >> 8
        inner ^= inner >> 4
        inner ^= inner >> 2
        inner ^= inner >> 1

        self.state[0] ^= ~inner & np.uint32(1)

    def advance(self, adv: np.uint32) -> None:
        """Advance SIMD-oriented Fast Mersenne Twister sequence by adv"""
        adv = np.uint32(adv)
        adv = (adv * np.uint32(2)) + np.uint32(self.index)
        while adv >= np.uint32(624):
            self.shuffle()
            adv -= np.uint32(624)
        self.index = np.uint16(adv)

    def next(self) -> np.uint64:
        """Access and return the next 64-bit rand"""
        if self.index == np.uint16(624):
            self.shuffle()

        low = self.state[self.index]
        self.index += 1
        high = self.state[self.index]
        self.index += 1
        return np.uint64(low) | (np.uint64(high) << np.uint64(32))

    def shuffle(self) -> None:
        """Advance and shuffle the entire state (624 advances)"""
        state = self.state

        # compute the first two states manually

        x_0 = state[0] << 8
        x_1 = (state[1] << 8) | (state[0] >> 24)
        x_2 = (state[2] << 8) | (state[1] >> 24)
        x_3 = (state[3] << 8) | (state[2] >> 24)

        y_0 = (state[617] << 24) | (state[616] >> 8)
        y_1 = (state[618] << 24) | (state[617] >> 8)
        y_2 = (state[619] << 24) | (state[618] >> 8)
        y_3 = state[619] >> 8

        b_0 = (state[488] >> 11) & 0xDFFFFFEF
        b_1 = (state[489] >> 11) & 0xDDFECB7F
        b_2 = (state[490] >> 11) & 0xBFFAFFFF
        b_3 = (state[491] >> 11) & 0xBFFFFFF6

        d_0 = state[620] << 18
        d_1 = state[621] << 18
        d_2 = state[622] << 18
        d_3 = state[623] << 18

        state[0] ^= x_0 ^ y_0 ^ b_0 ^ d_0
        state[1] ^= x_1 ^ y_1 ^ b_1 ^ d_1
        state[2] ^= x_2 ^ y_2 ^ b_2 ^ d_2
        state[3] ^= x_3 ^ y_3 ^ b_3 ^ d_3

        x_0 = state[4] << 8
        x_1 = (state[5] << 8) | (state[4] >> 24)
        x_2 = (state[6] << 8) | (state[5] >> 24)
        x_3 = (state[7] << 8) | (state[6] >> 24)

        y_0 = (state[621] << 24) | (state[620] >> 8)
        y_1 = (state[622] << 24) | (state[621] >> 8)
        y_2 = (state[623] << 24) | (state[622] >> 8)
        y_3 = state[623] >> 8

        b_0 = (state[492] >> 11) & 0xDFFFFFEF
        b_1 = (state[493] >> 11) & 0xDDFECB7F
        b_2 = (state[494] >> 11) & 0xBFFAFFFF
        b_3 = (state[495] >> 11) & 0xBFFFFFF6

        d_0 = state[0] << 18
        d_1 = state[1] << 18
        d_2 = state[2] << 18
        d_3 = state[3] << 18

        state[4] ^= x_0 ^ y_0 ^ b_0 ^ d_0
        state[5] ^= x_1 ^ y_1 ^ b_1 ^ d_1
        state[6] ^= x_2 ^ y_2 ^ b_2 ^ d_2
        state[7] ^= x_3 ^ y_3 ^ b_3 ^ d_3

        # TODO: shuffle should be somehow optimized to match
        # how sfmt is actually supposed to use simd

        for i in range(8, 136, 4):
            x_0 = state[i] << 8
            x_1 = (state[i + 1] << 8) | (state[i] >> 24)
            x_2 = (state[i + 2] << 8) | (state[i + 1] >> 24)
            x_3 = (state[i + 3] << 8) | (state[i + 2] >> 24)

            b_0 = (state[i + 488] >> 11) & 0xDFFFFFEF
            b_1 = (state[i + 489] >> 11) & 0xDDFECB7F
            b_2 = (state[i + 490] >> 11) & 0xBFFAFFFF
            b_3 = (state[i + 491] >> 11) & 0xBFFFFFF6

            d_0 = state[i - 4] << 18
            d_1 = state[i - 3] << 18
            d_2 = state[i - 2] << 18
            d_3 = state[i - 1] << 18

            y_0 = (state[i - 7] << 24) | (state[i - 8] >> 8)
            y_1 = (state[i - 6] << 24) | (state[i - 7] >> 8)
            y_2 = (state[i - 5] << 24) | (state[i - 6] >> 8)
            y_3 = state[i - 5] >> 8

            state[i] ^= x_0 ^ b_0 ^ d_0 ^ y_0
            state[i + 1] ^= x_1 ^ b_1 ^ d_1 ^ y_1
            state[i + 2] ^= x_2 ^ b_2 ^ d_2 ^ y_2
            state[i + 3] ^= x_3 ^ b_3 ^ d_3 ^ y_3

        for i in range(136, 624, 4):
            x_0 = state[i] << 8
            x_1 = (state[i + 1] << 8) | (state[i] >> 24)
            x_2 = (state[i + 2] << 8) | (state[i + 1] >> 24)
            x_3 = (state[i + 3] << 8) | (state[i + 2] >> 24)

            b_0 = (state[i - 136] >> 11) & 0xDFFFFFEF
            b_1 = (state[i - 135] >> 11) & 0xDDFECB7F
            b_2 = (state[i - 134] >> 11) & 0xBFFAFFFF
            b_3 = (state[i - 133] >> 11) & 0xBFFFFFF6

            d_0 = state[i - 4] << 18
            d_1 = state[i - 3] << 18
            d_2 = state[i - 2] << 18
            d_3 = state[i - 1] << 18

            y_0 = (state[i - 7] << 24) | (state[i - 8] >> 8)
            y_1 = (state[i - 6] << 24) | (state[i - 7] >> 8)
            y_2 = (state[i - 5] << 24) | (state[i - 6] >> 8)
            y_3 = state[i - 5] >> 8

            state[i] ^= x_0 ^ b_0 ^ d_0 ^ y_0
            state[i + 1] ^= x_1 ^ b_1 ^ d_1 ^ y_1
            state[i + 2] ^= x_2 ^ b_2 ^ d_2 ^ y_2
            state[i + 3] ^= x_3 ^ b_3 ^ d_3 ^ y_3

        self.index = 0

    def next_rand(self, maximum: np.uint64) -> np.uint64:
        """Generate and return the next [0, maximum) random uint via modulo distribution"""
        return self.next() % np.uint64(maximum)
