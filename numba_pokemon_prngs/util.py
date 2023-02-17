"""Utility @njit compiled numba functions"""
import numpy as np


def change_endian_u32(val: np.uint32) -> np.uint32:
    """Swap endian bytes"""
    val = ((val << np.uint32(8)) & np.uint32(0xFF00FF00)) | (
        (val >> np.uint32(8)) & np.uint32(0xFF00FF)
    )
    return (val << np.uint32(16)) | (val >> np.uint32(16))


def rotate_left_u32(val: np.uint32, count: np.uint8) -> np.uint32:
    """Rotate bits left by count"""
    return (val << count) | (val >> (np.uint8(32) - count))


def rotate_right_u32(val: np.uint32, count: np.uint8) -> np.uint32:
    """Rotate bits left by count"""
    return (val << (np.uint8(32) - count)) | (val >> count)
