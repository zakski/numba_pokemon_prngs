"""Personal Info specification for Pokemon Games"""

import importlib.resources as pkg_resources
import numpy as np
from .personal_info import (
    PersonalInfo1,
    PersonalInfo2,
    PersonalInfo3,
    PersonalInfo4,
    PersonalInfo5B2W2,
    PersonalInfo5BW,
    PersonalInfo6AO,
    PersonalInfo6XY,
    PersonalInfo7,
    PersonalInfo7GG,
    PersonalInfo8BDSP,
    PersonalInfo8LA,
    PersonalInfo8SWSH,
    PersonalInfo9SV,
)
from ...resources.bin import personal as personal_bin_directory


def load_personal(identifier: str, dtype: np.dtype) -> np.recarray:
    """Load array of PersonalInfo from file"""

    with pkg_resources.open_binary(
        personal_bin_directory, f"personal_{identifier}"
    ) as personal_info_file:
        return np.fromfile(personal_info_file, dtype=dtype).view(np.recarray)


PERSONAL_INFO_RB = load_personal("rb", PersonalInfo1)
PERSONAL_INFO_Y = load_personal("y", PersonalInfo1)

PERSONAL_INFO_GS = load_personal("gs", PersonalInfo2)
PERSONAL_INFO_C = load_personal("c", PersonalInfo2)

PERSONAL_INFO_RS = load_personal("rs", PersonalInfo3)
PERSONAL_INFO_E = load_personal("e", PersonalInfo3)
PERSONAL_INFO_FR = load_personal("fr", PersonalInfo3)
PERSONAL_INFO_LG = load_personal("lg", PersonalInfo3)

PERSONAL_INFO_DP = load_personal("dp", PersonalInfo4)
PERSONAL_INFO_PT = load_personal("pt", PersonalInfo4)

PERSONAL_INFO_BW = load_personal("bw", PersonalInfo5BW)
PERSONAL_INFO_B2W2 = load_personal("b2w2", PersonalInfo5B2W2)

PERSONAL_INFO_XY = load_personal("xy", PersonalInfo6XY)
PERSONAL_INFO_ORAS = load_personal("ao", PersonalInfo6AO)

PERSONAL_INFO_SM = load_personal("sm", PersonalInfo7)
PERSONAL_INFO_USUM = load_personal("uu", PersonalInfo7)

PERSONAL_INFO_LGPE = load_personal("gg", PersonalInfo7GG)

PERSONAL_INFO_SWSH = load_personal("swsh", PersonalInfo8SWSH)
PERSONAL_INFO_BDSP = load_personal("bdsp", PersonalInfo8BDSP)
PERSONAL_INFO_LA = load_personal("la", PersonalInfo8LA)

PERSONAL_INFO_SV = load_personal("sv", PersonalInfo9SV)
