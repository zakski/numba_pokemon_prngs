"""Utility functions for the data submodule"""

from dataclasses import dataclass, fields
import numpy as np

U8 = np.dtype("u1")
U16 = np.dtype("<u2")
U32 = np.dtype("<u4")


def dtype_dataclass(cls) -> np.dtype:
    """Decorator to turn a dataclass into a numpy dtype"""
    dataclass_cls = dataclass(cls)
    return np.dtype([(field.name, field.type) for field in fields(dataclass_cls)])


def unused_bytes(size: int) -> np.dtype:
    """Numpy dtype for an unused byte array"""
    return np.dtype([("", U8, (size,))])
