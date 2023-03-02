"""PersonalInfo for each Generation"""

import numpy as np
from ..util import dtype_dataclass, unused_bytes, U8


@dtype_dataclass
class PersonalInfo1:
    """PersonalInfo for Generation 1"""

    gender: U8
    hp: U8
    attack: U8
    defense: U8
    speed: U8
    special: U8
    type_1: U8
    type_2: U8
    catch_rate: U8
    base_exp: U8
    _padding: unused_bytes(6)
    move_1: U8
    move_2: U8
    move_3: U8
    move_4: U8
    exp_growth: U8
    tm_hm: unused_bytes(7)


PersonalInfo1: np.dtype
assert PersonalInfo1.itemsize == 0x1C


@dtype_dataclass
class PersonalInfo2:
    """PersonalInfo for Generation 2"""

    dex_id: U8
    hp: U8
    attack: U8
    defense: U8
    speed: U8
    special_attack: U8
    special_defense: U8
    type_1: U8
    type_2: U8
    catch_rate: U8
    base_exp: U8
    item_1: U8
    item_2: U8
    gender: U8
    hatch_cycles: U8
    _padding: unused_bytes(7)
    exp_growth: U8
    egg_group: U8
    tm_hm: unused_bytes(8)


PersonalInfo2: np.dtype
assert PersonalInfo2.itemsize == 0x20


# TODO: G3+ personal info specification


@dtype_dataclass
class PersonalInfo3:
    """PersonalInfo for Generation 3"""

    data: unused_bytes(0x1C)


PersonalInfo3: np.dtype
assert PersonalInfo3.itemsize == 0x1C


@dtype_dataclass
class PersonalInfo4:
    """PersonalInfo for Generation 4"""

    data: unused_bytes(0x2C)


PersonalInfo4: np.dtype
assert PersonalInfo4.itemsize == 0x2C


@dtype_dataclass
class PersonalInfo5BW:
    """PersonalInfo for Generation 5 Black and White"""

    data: unused_bytes(0x3C)


PersonalInfo5BW: np.dtype
assert PersonalInfo5BW.itemsize == 0x3C


@dtype_dataclass
class PersonalInfo5B2W2:
    """PersonalInfo for Generation 5 Black 2 and White 2"""

    data: unused_bytes(0x4C)


PersonalInfo5B2W2: np.dtype
assert PersonalInfo5B2W2.itemsize == 0x4C


@dtype_dataclass
class PersonalInfo6XY:
    """PersonalInfo for Generation 6 X and Y"""

    data: unused_bytes(0x40)


PersonalInfo6XY: np.dtype
assert PersonalInfo6XY.itemsize == 0x40


@dtype_dataclass
class PersonalInfo6AO:
    """PersonalInfo for Generation 6 Omega Ruby and Alpha Sapphire"""

    data: unused_bytes(0x50)


PersonalInfo6AO: np.dtype
assert PersonalInfo6AO.itemsize == 0x50


@dtype_dataclass
class PersonalInfo7:
    """PersonalInfo for Generation 7"""

    data: unused_bytes(0x54)


PersonalInfo7: np.dtype
assert PersonalInfo7.itemsize == 0x54


@dtype_dataclass
class PersonalInfo7GG:
    """PersonalInfo for Let's Go Pikachu and Eevee"""

    data: unused_bytes(0x54)


PersonalInfo7GG: np.dtype
assert PersonalInfo7GG.itemsize == 0x54


@dtype_dataclass
class PersonalInfo8SWSH:
    """PersonalInfo for Generation 8 Sword and Shield"""

    data: unused_bytes(0xB0)


PersonalInfo8SWSH: np.dtype
assert PersonalInfo8SWSH.itemsize == 0xB0


@dtype_dataclass
class PersonalInfo8BDSP:
    """PersonalInfo for Generation 8 Brilliant Diamond and Shining Pearl"""

    data: unused_bytes(0x44)


PersonalInfo8BDSP: np.dtype
assert PersonalInfo8BDSP.itemsize == 0x44


@dtype_dataclass
class PersonalInfo8LA:
    """PersonalInfo for Generation 8 Pokemon: Legends Arceus"""

    data: unused_bytes(0xB0)


PersonalInfo8LA: np.dtype
assert PersonalInfo8LA.itemsize == 0xB0


@dtype_dataclass
class PersonalInfo9SV:
    """PersonalInfo for Generation 9 Scarlet and Violet"""

    data: unused_bytes(0x44)


PersonalInfo9SV: np.dtype
assert PersonalInfo9SV.itemsize == 0x44
