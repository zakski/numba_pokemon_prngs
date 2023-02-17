"""SHA1 hash for Gen 5 initial seed generation"""

from __future__ import annotations
import numba
import numpy as np
from .lcrng import BWRNG
from .enums import Game, Language, DSType
from .util import change_endian_u32, rotate_left_u32, rotate_right_u32

# "nazos" are game version and language specific constants that happen to be stored next to
# SHA1 hashed data
@numba.njit(numba.uint32[:](numba.uint32))
def compute_nazo_bw(input_nazo: np.uint32) -> np.ndarray[np.uint32, 5]:
    """Compute the "nazo" values for black and white"""
    nazos = np.empty(5, dtype=np.uint32)
    nazos[0] = change_endian_u32(input_nazo)
    nazos[1] = nazos[2] = change_endian_u32(input_nazo + np.uint32(0xFC))
    nazos[3] = nazos[4] = change_endian_u32(
        input_nazo + np.uint32(0xFC) + np.uint32(0x4C)
    )

    return nazos


@numba.njit(numba.uint32[:](numba.uint32, numba.uint32, numba.uint32))
def compute_nazo_bw2(
    input_nazo: np.uint32, input_nazo0: np.uint32, input_nazo1: np.uint32
) -> np.ndarray[np.uint32, 5]:
    """Compute the "nazo" values for black 2 and white 2"""
    nazos = np.empty(5, dtype=np.uint32)
    nazos[0] = change_endian_u32(input_nazo0)
    nazos[1] = change_endian_u32(input_nazo1)
    nazos[2] = change_endian_u32(input_nazo)
    nazos[3] = nazos[4] = change_endian_u32(input_nazo + np.uint32(0x54))

    return nazos


ENGLISH_BLACK = compute_nazo_bw(0x022160B0)
ENGLISH_WHITE = compute_nazo_bw(0x022160D0)
ENGLISH_BLACK_DSI = compute_nazo_bw(0x02760190)
ENGLISH_WHITE_DSI = compute_nazo_bw(0x027601B0)
ENGLISH_BLACK2 = compute_nazo_bw2(0x02200010, 0x0209AEE8, 0x02039DE9)
ENGLISH_WHITE2 = compute_nazo_bw2(0x02200050, 0x0209AF28, 0x02039E15)
ENGLISH_BLACK2_DSI = compute_nazo_bw2(0x027A5F70, 0x0209AEE8, 0x02039DE9)
ENGLISH_WHITE2_DSI = compute_nazo_bw2(0x027A5E90, 0x0209AF28, 0x02039E15)

JAPANESE_BLACK = compute_nazo_bw(0x02215F10)
JAPANESE_WHITE = compute_nazo_bw(0x02215F30)
JAPANESE_BLACK_DSI = compute_nazo_bw(0x02761150)
JAPANESE_WHITE_DSI = compute_nazo_bw(0x02761150)
JAPANESE_BLACK2 = compute_nazo_bw2(0x021FF9B0, 0x0209A8DC, 0x02039AC9)
JAPANESE_WHITE2 = compute_nazo_bw2(0x021FF9D0, 0x0209A8FC, 0x02039AF5)
JAPANESE_BLACK2_DSI = compute_nazo_bw2(0x027AA730, 0x0209A8DC, 0x02039AC9)
JAPANESE_WHITE2_DSI = compute_nazo_bw2(0x027AA5F0, 0x0209A8FC, 0x02039AF5)

GERMAN_BLACK = compute_nazo_bw(0x02215FF0)
GERMAN_WHITE = compute_nazo_bw(0x02216010)
GERMAN_BLACK_DSI = compute_nazo_bw(0x027602F0)
GERMAN_WHITE_DSI = compute_nazo_bw(0x027602F0)
GERMAN_BLACK2 = compute_nazo_bw2(0x021FFF50, 0x0209AE28, 0x02039D69)
GERMAN_WHITE2 = compute_nazo_bw2(0x021FFF70, 0x0209AE48, 0x02039D95)
GERMAN_BLACK2_DSI = compute_nazo_bw2(0x027A6110, 0x0209AE28, 0x02039D69)
GERMAN_WHITE2_DSI = compute_nazo_bw2(0x027A6010, 0x0209AE48, 0x02039D95)

SPANISH_BLACK = compute_nazo_bw(0x02216070)
SPANISH_WHITE = compute_nazo_bw(0x02216070)
SPANISH_BLACK_DSI = compute_nazo_bw(0x027601F0)
SPANISH_WHITE_DSI = compute_nazo_bw(0x027601F0)
SPANISH_BLACK2 = compute_nazo_bw2(0x021FFFD0, 0x0209AEA8, 0x02039DB9)
SPANISH_WHITE2 = compute_nazo_bw2(0x021FFFF0, 0x0209AEC8, 0x02039DE5)
SPANISH_BLACK2_DSI = compute_nazo_bw2(0x027A6070, 0x0209AEA8, 0x02039DB9)
SPANISH_WHITE2_DSI = compute_nazo_bw2(0x027A5FB0, 0x0209AEC8, 0x02039DE5)

FRENCH_BLACK = compute_nazo_bw(0x02216030)
FRENCH_WHITE = compute_nazo_bw(0x02216050)
FRENCH_BLACK_DSI = compute_nazo_bw(0x02760230)
FRENCH_WHITE_DSI = compute_nazo_bw(0x02760250)
FRENCH_BLACK2 = compute_nazo_bw2(0x02200030, 0x0209AF08, 0x02039DF9)
FRENCH_WHITE2 = compute_nazo_bw2(0x02200050, 0x0209AF28, 0x02039E25)
FRENCH_BLACK2_DSI = compute_nazo_bw2(0x027A5F90, 0x0209AF08, 0x02039DF9)
FRENCH_WHITE2_DSI = compute_nazo_bw2(0x027A5EF0, 0x0209AF28, 0x02039E25)

ITALIAN_BLACK = compute_nazo_bw(0x02215FB0)
ITALIAN_WHITE = compute_nazo_bw(0x02215FD0)
ITALIAN_BLACK_DSI = compute_nazo_bw(0x027601D0)
ITALIAN_WHITE_DSI = compute_nazo_bw(0x027601D0)
ITALIAN_BLACK2 = compute_nazo_bw2(0x021FFF10, 0x0209ADE8, 0x02039D69)
ITALIAN_WHITE2 = compute_nazo_bw2(0x021FFF50, 0x0209AE28, 0x02039D95)
ITALIAN_BLACK2_DSI = compute_nazo_bw2(0x027A5F70, 0x0209ADE8, 0x02039D69)
ITALIAN_WHITE2_DSI = compute_nazo_bw2(0x027A5ED0, 0x0209AE28, 0x02039D95)

KOREAN_BLACK = compute_nazo_bw(0x022167B0)
KOREAN_WHITE = compute_nazo_bw(0x022167B0)
KOREAN_BLACK_DSI = compute_nazo_bw(0x02761150)
KOREAN_WHITE_DSI = compute_nazo_bw(0x02761150)
KOREAN_BLACK2 = compute_nazo_bw2(0x02200750, 0x0209B60C, 0x0203A4D5)
KOREAN_WHITE2 = compute_nazo_bw2(0x02200770, 0x0209B62C, 0x0203A501)
KOREAN_BLACK2_DSI = compute_nazo_bw2(0x02200770, 0x0209B60C, 0x0203A4D5)
KOREAN_WHITE2_DSI = compute_nazo_bw2(0x027A57B0, 0x0209B62C, 0x0203A501)


@numba.njit(
    numba.types.Array(numba.uint32, 1, "C", readonly=True)(
        numba.uint32, numba.uint8, numba.uint8
    )
)
def get_nazo(
    version: Game, language: Language, ds_type: DSType
) -> np.ndarray[np.uint32, 5]:
    """Get the nazos for a specific game, language, and ds type"""
    if language == Language.ENGLISH:
        if version == Game.BLACK:
            return ENGLISH_BLACK if ds_type == DSType.DS else ENGLISH_BLACK_DSI
        if version == Game.WHITE:
            return ENGLISH_WHITE if ds_type == DSType.DS else ENGLISH_WHITE_DSI
        if version == Game.BLACK2:
            return ENGLISH_BLACK2 if ds_type == DSType.DS else ENGLISH_BLACK2_DSI
        if version == Game.WHITE2:
            return ENGLISH_WHITE2 if ds_type == DSType.DS else ENGLISH_WHITE2_DSI
    if language == Language.JAPANESE:
        if version == Game.BLACK:
            return JAPANESE_BLACK if ds_type == DSType.DS else JAPANESE_BLACK_DSI
        if version == Game.WHITE:
            return JAPANESE_WHITE if ds_type == DSType.DS else JAPANESE_WHITE_DSI
        if version == Game.BLACK2:
            return JAPANESE_BLACK2 if ds_type == DSType.DS else JAPANESE_BLACK2_DSI
        if version == Game.WHITE2:
            return JAPANESE_WHITE2 if ds_type == DSType.DS else JAPANESE_WHITE2_DSI
    if language == Language.GERMAN:
        if version == Game.BLACK:
            return GERMAN_BLACK if ds_type == DSType.DS else GERMAN_BLACK_DSI
        if version == Game.WHITE:
            return GERMAN_WHITE if ds_type == DSType.DS else GERMAN_WHITE_DSI
        if version == Game.BLACK2:
            return GERMAN_BLACK2 if ds_type == DSType.DS else GERMAN_BLACK2_DSI
        if version == Game.WHITE2:
            return GERMAN_WHITE2 if ds_type == DSType.DS else GERMAN_WHITE2_DSI
    if language == Language.SPANISH:
        if version == Game.BLACK:
            return SPANISH_BLACK if ds_type == DSType.DS else SPANISH_BLACK_DSI
        if version == Game.WHITE:
            return SPANISH_WHITE if ds_type == DSType.DS else SPANISH_WHITE_DSI
        if version == Game.BLACK2:
            return SPANISH_BLACK2 if ds_type == DSType.DS else SPANISH_BLACK2_DSI
        if version == Game.WHITE2:
            return SPANISH_WHITE2 if ds_type == DSType.DS else SPANISH_WHITE2_DSI
    if language == Language.FRENCH:
        if version == Game.BLACK:
            return FRENCH_BLACK if ds_type == DSType.DS else FRENCH_BLACK_DSI
        if version == Game.WHITE:
            return FRENCH_WHITE if ds_type == DSType.DS else FRENCH_WHITE_DSI
        if version == Game.BLACK2:
            return FRENCH_BLACK2 if ds_type == DSType.DS else FRENCH_BLACK2_DSI
        if version == Game.WHITE2:
            return FRENCH_WHITE2 if ds_type == DSType.DS else FRENCH_WHITE2_DSI
    if language == Language.ITALIAN:
        if version == Game.BLACK:
            return ITALIAN_BLACK if ds_type == DSType.DS else ITALIAN_BLACK_DSI
        if version == Game.WHITE:
            return ITALIAN_WHITE if ds_type == DSType.DS else ITALIAN_WHITE_DSI
        if version == Game.BLACK2:
            return ITALIAN_BLACK2 if ds_type == DSType.DS else ITALIAN_BLACK2_DSI
        if version == Game.WHITE2:
            return ITALIAN_WHITE2 if ds_type == DSType.DS else ITALIAN_WHITE2_DSI
    if language == Language.KOREAN:
        if version == Game.BLACK:
            return KOREAN_BLACK if ds_type == DSType.DS else KOREAN_BLACK_DSI
        if version == Game.WHITE:
            return KOREAN_WHITE if ds_type == DSType.DS else KOREAN_WHITE_DSI
        if version == Game.BLACK2:
            return KOREAN_BLACK2 if ds_type == DSType.DS else KOREAN_BLACK2_DSI
        if version == Game.WHITE2:
            return KOREAN_WHITE2 if ds_type == DSType.DS else KOREAN_WHITE2_DSI
    return np.zeros(5, dtype=np.uint32)


BCD = np.array(
    tuple(((i // 10) << 4) | ((i % 10)) for i in range(100)),
    dtype=np.uint32,
)


@numba.njit(
    numba.types.UniTuple(numba.uint32, 2)(
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
    ),
    locals={"t_val": numba.uint32},
    inline="always",
)
def section1_calc(
    a_val: np.uint32,
    b_val: np.uint32,
    c_val: np.uint32,
    d_val: np.uint32,
    e_val: np.uint32,
    input_val: np.uint32,
) -> tuple[np.uint32, np.uint32]:
    """Hash calc for section 1: 0-19"""
    t_val = np.uint32(
        rotate_left_u32(a_val, 5)
        + ((b_val & c_val) | (~b_val & d_val))
        + e_val
        + 0x5A827999
        + input_val
    )
    b_val = rotate_right_u32(b_val, 2)
    return t_val, b_val


@numba.njit(
    numba.types.UniTuple(numba.uint32, 2)(
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
    ),
    locals={"t_val": numba.uint32},
    inline="always",
)
def section2_calc(
    a_val: np.uint32,
    b_val: np.uint32,
    c_val: np.uint32,
    d_val: np.uint32,
    e_val: np.uint32,
    input_val: np.uint32,
) -> tuple[np.uint32, numba.uint32]:
    """Hash calc for section 2: 20-39"""
    t_val = (
        rotate_left_u32(a_val, 5)
        + (b_val ^ c_val ^ d_val)
        + e_val
        + 0x6ED9EBA1
        + input_val
    )
    b_val = rotate_right_u32(b_val, 2)
    return t_val, b_val


@numba.njit(
    numba.types.UniTuple(numba.uint32, 2)(
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
    ),
    locals={"t_val": numba.uint32},
    inline="always",
)
def section3_calc(
    a_val: np.uint32,
    b_val: np.uint32,
    c_val: np.uint32,
    d_val: np.uint32,
    e_val: np.uint32,
    input_val: np.uint32,
) -> tuple[np.uint32, np.uint32]:
    """Hash calc for section 3: 40-59"""
    t_val = (
        rotate_left_u32(a_val, 5)
        + ((b_val & c_val) | ((b_val | c_val) & d_val))
        + e_val
        + 0x8F1BBCDC
        + input_val
    )
    b_val = rotate_right_u32(b_val, 2)
    return t_val, b_val


@numba.njit(
    numba.types.UniTuple(numba.uint32, 2)(
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
        numba.uint32,
    ),
    locals={"t_val": numba.uint32},
    inline="always",
)
def section4_calc(
    a_val: np.uint32,
    b_val: np.uint32,
    c_val: np.uint32,
    d_val: np.uint32,
    e_val: np.uint32,
    input_val: np.uint32,
) -> tuple[np.uint32, np.uint32]:
    """Hash calc for section 4: 60-79"""
    t_val = (
        rotate_left_u32(a_val, 5)
        + (b_val ^ c_val ^ d_val)
        + e_val
        + 0xCA62C1D6
        + input_val
    )
    b_val = rotate_right_u32(b_val, 2)
    return t_val, b_val


@numba.experimental.jitclass
class SHA1:
    """SHA1 hash for Gen 5 initial seed generation"""

    data: numba.uint32[::1]  # contiguous array

    def __init__(
        self,
        version: Game,
        language: Language,
        ds_type: DSType,
        mac: np.uint64,
        soft_reset: bool,
        v_frame: np.uint8,
        gx_state: np.uint8,
    ) -> None:
        self.data = np.empty(80, dtype=np.uint32)
        data = self.data
        data[:5] = get_nazo(version, language, ds_type)
        data[6] = mac & np.uint32(0xFFFF)
        if soft_reset:
            data[6] ^= np.uint32(0x1000000)
        data[7] = (
            np.uint32(mac >> np.uint64(16))
            ^ (np.uint32(v_frame) << np.uint32(24))
            ^ np.uint32(gx_state)
        )

        # set values
        data[10] = 0
        data[11] = 0
        data[13] = 0x80000000
        data[14] = 0
        data[15] = 0x1A0

        # precompute data[18]
        data[18] = rotate_left_u32(data[15] ^ data[10] ^ data[4] ^ data[2], 1)

    def hash_seed(self, alpha: np.ndarray[np.uint32, 5]) -> np.uint64:
        """Compute hash based on precomputed alpha"""
        data = self.data
        a_val = alpha[0]
        b_val = alpha[1]
        c_val = alpha[2]
        d_val = alpha[3]
        e_val = alpha[4]
        self.calc_w(17)
        self.calc_w(20)
        self.calc_w(23)
        self.calc_w(25)
        self.calc_w(26)
        self.calc_w(28)
        self.calc_w(29)
        self.calc_w(31)
        for i in range(32, 80):
            self.calc_w_simd(i)

        t_val, b_val = section1_calc(a_val, b_val, c_val, d_val, e_val, data[9])
        e_val, a_val = section1_calc(t_val, a_val, b_val, c_val, d_val, data[10])
        d_val, t_val = section1_calc(e_val, t_val, a_val, b_val, c_val, data[11])
        c_val, e_val = section1_calc(d_val, e_val, t_val, a_val, b_val, data[12])
        b_val, d_val = section1_calc(c_val, d_val, e_val, t_val, a_val, data[13])
        a_val, c_val = section1_calc(b_val, c_val, d_val, e_val, t_val, data[14])
        t_val, b_val = section1_calc(a_val, b_val, c_val, d_val, e_val, data[15])
        e_val, a_val = section1_calc(t_val, a_val, b_val, c_val, d_val, data[16])
        d_val, t_val = section1_calc(e_val, t_val, a_val, b_val, c_val, data[17])
        c_val, e_val = section1_calc(d_val, e_val, t_val, a_val, b_val, data[18])
        b_val, d_val = section1_calc(c_val, d_val, e_val, t_val, a_val, data[19])

        a_val, c_val = section2_calc(b_val, c_val, d_val, e_val, t_val, data[20])
        t_val, b_val = section2_calc(a_val, b_val, c_val, d_val, e_val, data[21])
        e_val, a_val = section2_calc(t_val, a_val, b_val, c_val, d_val, data[22])
        d_val, t_val = section2_calc(e_val, t_val, a_val, b_val, c_val, data[23])
        c_val, e_val = section2_calc(d_val, e_val, t_val, a_val, b_val, data[24])
        b_val, d_val = section2_calc(c_val, d_val, e_val, t_val, a_val, data[25])
        a_val, c_val = section2_calc(b_val, c_val, d_val, e_val, t_val, data[26])
        t_val, b_val = section2_calc(a_val, b_val, c_val, d_val, e_val, data[27])
        e_val, a_val = section2_calc(t_val, a_val, b_val, c_val, d_val, data[28])
        d_val, t_val = section2_calc(e_val, t_val, a_val, b_val, c_val, data[29])
        c_val, e_val = section2_calc(d_val, e_val, t_val, a_val, b_val, data[30])
        b_val, d_val = section2_calc(c_val, d_val, e_val, t_val, a_val, data[31])
        a_val, c_val = section2_calc(b_val, c_val, d_val, e_val, t_val, data[32])
        t_val, b_val = section2_calc(a_val, b_val, c_val, d_val, e_val, data[33])
        e_val, a_val = section2_calc(t_val, a_val, b_val, c_val, d_val, data[34])
        d_val, t_val = section2_calc(e_val, t_val, a_val, b_val, c_val, data[35])
        c_val, e_val = section2_calc(d_val, e_val, t_val, a_val, b_val, data[36])
        b_val, d_val = section2_calc(c_val, d_val, e_val, t_val, a_val, data[37])
        a_val, c_val = section2_calc(b_val, c_val, d_val, e_val, t_val, data[38])
        t_val, b_val = section2_calc(a_val, b_val, c_val, d_val, e_val, data[39])

        e_val, a_val = section3_calc(t_val, a_val, b_val, c_val, d_val, data[40])
        d_val, t_val = section3_calc(e_val, t_val, a_val, b_val, c_val, data[41])
        c_val, e_val = section3_calc(d_val, e_val, t_val, a_val, b_val, data[42])
        b_val, d_val = section3_calc(c_val, d_val, e_val, t_val, a_val, data[43])
        a_val, c_val = section3_calc(b_val, c_val, d_val, e_val, t_val, data[44])
        t_val, b_val = section3_calc(a_val, b_val, c_val, d_val, e_val, data[45])
        e_val, a_val = section3_calc(t_val, a_val, b_val, c_val, d_val, data[46])
        d_val, t_val = section3_calc(e_val, t_val, a_val, b_val, c_val, data[47])
        c_val, e_val = section3_calc(d_val, e_val, t_val, a_val, b_val, data[48])
        b_val, d_val = section3_calc(c_val, d_val, e_val, t_val, a_val, data[49])
        a_val, c_val = section3_calc(b_val, c_val, d_val, e_val, t_val, data[50])
        t_val, b_val = section3_calc(a_val, b_val, c_val, d_val, e_val, data[51])
        e_val, a_val = section3_calc(t_val, a_val, b_val, c_val, d_val, data[52])
        d_val, t_val = section3_calc(e_val, t_val, a_val, b_val, c_val, data[53])
        c_val, e_val = section3_calc(d_val, e_val, t_val, a_val, b_val, data[54])
        b_val, d_val = section3_calc(c_val, d_val, e_val, t_val, a_val, data[55])
        a_val, c_val = section3_calc(b_val, c_val, d_val, e_val, t_val, data[56])
        t_val, b_val = section3_calc(a_val, b_val, c_val, d_val, e_val, data[57])
        e_val, a_val = section3_calc(t_val, a_val, b_val, c_val, d_val, data[58])
        d_val, t_val = section3_calc(e_val, t_val, a_val, b_val, c_val, data[59])

        c_val, e_val = section4_calc(d_val, e_val, t_val, a_val, b_val, data[60])
        b_val, d_val = section4_calc(c_val, d_val, e_val, t_val, a_val, data[61])
        a_val, c_val = section4_calc(b_val, c_val, d_val, e_val, t_val, data[62])
        t_val, b_val = section4_calc(a_val, b_val, c_val, d_val, e_val, data[63])
        e_val, a_val = section4_calc(t_val, a_val, b_val, c_val, d_val, data[64])
        d_val, t_val = section4_calc(e_val, t_val, a_val, b_val, c_val, data[65])
        c_val, e_val = section4_calc(d_val, e_val, t_val, a_val, b_val, data[66])
        b_val, d_val = section4_calc(c_val, d_val, e_val, t_val, a_val, data[67])
        a_val, c_val = section4_calc(b_val, c_val, d_val, e_val, t_val, data[68])
        t_val, b_val = section4_calc(a_val, b_val, c_val, d_val, e_val, data[69])
        e_val, a_val = section4_calc(t_val, a_val, b_val, c_val, d_val, data[70])
        d_val, t_val = section4_calc(e_val, t_val, a_val, b_val, c_val, data[71])
        c_val, e_val = section4_calc(d_val, e_val, t_val, a_val, b_val, data[72])
        b_val, d_val = section4_calc(c_val, d_val, e_val, t_val, a_val, data[73])
        a_val, c_val = section4_calc(b_val, c_val, d_val, e_val, t_val, data[74])
        t_val, b_val = section4_calc(a_val, b_val, c_val, d_val, e_val, data[75])
        e_val, a_val = section4_calc(t_val, a_val, b_val, c_val, d_val, data[76])
        d_val, t_val = section4_calc(e_val, t_val, a_val, b_val, c_val, data[77])
        c_val, e_val = section4_calc(d_val, e_val, t_val, a_val, b_val, data[78])
        b_val, d_val = section4_calc(c_val, d_val, e_val, t_val, a_val, data[79])

        part1 = change_endian_u32(b_val + 0x67452301)
        part2 = change_endian_u32(c_val + 0xEFCDAB89)

        seed = np.uint64(
            np.uint64(np.uint64(part2) << np.uint64(32)) | np.uint64(part1)
        )

        return BWRNG(seed).next()

    def precompute(self) -> np.ndarray[np.uint32, 5]:
        """Precompute alpha values"""
        data = self.data

        a_val = 0x67452301
        b_val = 0xEFCDAB89
        c_val = 0x98BADCFE
        d_val = 0x10325476
        e_val = 0xC3D2E1F0

        t_val, b_val = section1_calc(a_val, b_val, c_val, d_val, e_val, data[0])
        e_val, a_val = section1_calc(t_val, a_val, b_val, c_val, d_val, data[1])
        d_val, t_val = section1_calc(e_val, t_val, a_val, b_val, c_val, data[2])
        c_val, e_val = section1_calc(d_val, e_val, t_val, a_val, b_val, data[3])
        b_val, d_val = section1_calc(c_val, d_val, e_val, t_val, a_val, data[4])
        a_val, c_val = section1_calc(b_val, c_val, d_val, e_val, t_val, data[5])
        t_val, b_val = section1_calc(a_val, b_val, c_val, d_val, e_val, data[6])
        e_val, a_val = section1_calc(t_val, a_val, b_val, c_val, d_val, data[7])
        d_val, t_val = section1_calc(e_val, t_val, a_val, b_val, c_val, data[8])

        self.calc_w(16)
        # self.calc_w(18) precomputed
        self.calc_w(19)
        self.calc_w(21)
        self.calc_w(22)
        self.calc_w(24)
        self.calc_w(27)
        self.calc_w(30)

        alpha = np.empty(5, dtype=np.uint32)
        alpha[0] = d_val
        alpha[1] = e_val
        alpha[2] = t_val
        alpha[3] = a_val
        alpha[4] = b_val

        return alpha

    def set_button(self, button: np.uint32) -> None:
        """Set held button"""
        self.data[12] = button

    def set_date(
        self, year: np.uint16, month: np.uint8, day: np.uint8, day_of_week: np.uint8
    ):
        """Set start up date"""
        self.data[8] = (
            np.uint32(np.uint32(BCD[year - 2000]) << np.uint32(24))
            | np.uint32(np.uint32(BCD[month]) << np.uint32(16))
            | np.uint32(np.uint32(BCD[day]) << np.uint32(8))
            | np.uint32(day_of_week)
        )

    def set_timer0(self, timer0: np.uint32, vcount: np.uint8) -> None:
        """Set Timer0 and vcount value"""
        self.data[5] = change_endian_u32(
            np.uint32(np.uint32(np.uint32(vcount) << np.uint(16)) | np.uint32(timer0))
        )

    def set_time(
        self, hour: np.uint8, minute: np.uint8, second: np.uint8, ds_type: DSType
    ):
        """Set start up time"""
        h_val = np.uint32(
            np.uint32(BCD[hour])
            + (
                np.uint32(0x40)
                if hour >= 12 and ds_type != DSType.DS3
                else np.uint32(0)
            )
        ) << np.uint32(24)
        m_val = np.uint32(np.uint32(BCD[minute]) << np.uint32(16))
        s_val = np.uint32(np.uint32(BCD[second]) << np.uint32(8))
        self.data[9] = h_val | m_val | s_val

    def calc_w(self, i: np.uint32) -> np.uint32:
        """Calc hash input value"""
        data = self.data
        val = rotate_left_u32(
            data[i - 3] ^ data[i - 8] ^ data[i - 14] ^ data[i - 16], 1
        )
        data[i] = val
        return val

    def calc_w_simd(self, i: np.uint32) -> np.uint32:
        """Calc hash input value w/SIMD"""
        data = self.data
        val = rotate_left_u32(
            data[i - 6] ^ data[i - 16] ^ data[i - 28] ^ data[i - 32], 2
        )
        data[i] = int(val)
        return val
