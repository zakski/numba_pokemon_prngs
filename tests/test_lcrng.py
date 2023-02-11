"""Tests for LCRNG classes"""
from numba_pokemon_prngs.lcrng import (
    PokeRNGDiv,
    PokeRNGMod,
    PokeRNGRDiv,
    PokeRNGRMod,
    ARNG,
    ARNGR,
    XDRNG,
    XDRNGR,
)


def test_lcrng_next():
    """Test LCRNG next() calls for full seed"""
    test_pokerng_div = PokeRNGDiv(0x12345678)
    test_pokerng_mod = PokeRNGMod(0x12345678)
    test_arng = ARNG(0x12345678)
    test_xdrng = XDRNG(0x12345678)

    test_pokerngr_div = PokeRNGRDiv(0x12345678)
    test_pokerngr_mod = PokeRNGRMod(0x12345678)
    test_arngr = ARNGR(0x12345678)
    test_xdrngr = XDRNGR(0x12345678)
    assert tuple(test_pokerng_div.next() for _ in range(5)) == (
        192004491,
        2229936802,
        3649731437,
        4108329948,
        646229279,
    )
    assert tuple(test_pokerng_mod.next() for _ in range(5)) == (
        192004491,
        2229936802,
        3649731437,
        4108329948,
        646229279,
    )
    assert tuple(test_arng.next() for _ in range(5)) == (
        775181657,
        499207454,
        4082793175,
        2119206356,
        103760037,
    )
    assert tuple(test_xdrng.next() for _ in range(5)) == (
        3018423131,
        1530878130,
        3423460525,
        3768712380,
        1536596111,
    )

    assert tuple(test_pokerngr_div.next() for _ in range(5)) == (
        1358145273,
        3431847134,
        687875895,
        3556988756,
        4100256197,
    )
    assert tuple(test_pokerngr_mod.next() for _ in range(5)) == (
        1358145273,
        3431847134,
        687875895,
        3556988756,
        4100256197,
    )
    assert tuple(test_arngr.next() for _ in range(5)) == (
        2408337579,
        2286114914,
        3692372301,
        373632348,
        1342820287,
    )
    assert tuple(test_xdrngr.next() for _ in range(5)) == (
        3745948697,
        399063950,
        1307699815,
        631669108,
        1486693829,
    )


def test_lcrng_next_u16():
    """Test LCRNG next_u16() calls for 16-bit rand"""
    test_pokerng_div = PokeRNGDiv(0x12345678)
    test_pokerng_mod = PokeRNGMod(0x12345678)
    test_arng = ARNG(0x12345678)
    test_xdrng = XDRNG(0x12345678)

    test_pokerngr_div = PokeRNGRDiv(0x12345678)
    test_pokerngr_mod = PokeRNGRMod(0x12345678)
    test_arngr = ARNGR(0x12345678)
    test_xdrngr = XDRNGR(0x12345678)
    assert tuple(test_pokerng_div.next_u16() for _ in range(5)) == (
        2929,
        34026,
        55690,
        62688,
        9860,
    )
    assert tuple(test_pokerng_mod.next_u16() for _ in range(5)) == (
        2929,
        34026,
        55690,
        62688,
        9860,
    )
    assert tuple(test_arng.next_u16() for _ in range(5)) == (
        11828,
        7617,
        62298,
        32336,
        1583,
    )
    assert tuple(test_xdrng.next_u16() for _ in range(5)) == (
        46057,
        23359,
        52237,
        57505,
        23446,
    )

    assert tuple(test_pokerngr_div.next_u16() for _ in range(5)) == (
        20723,
        52365,
        10496,
        54275,
        62564,
    )
    assert tuple(test_pokerngr_mod.next_u16() for _ in range(5)) == (
        20723,
        52365,
        10496,
        54275,
        62564,
    )
    assert tuple(test_arngr.next_u16() for _ in range(5)) == (
        36748,
        34883,
        56341,
        5701,
        20489,
    )
    assert tuple(test_xdrngr.next_u16() for _ in range(5)) == (
        57158,
        6089,
        19953,
        9638,
        22685,
    )


def test_lcrng_next_rand():
    """Test LCRNG next_rand() calls for constrained rand"""
    test_pokerng_div = PokeRNGDiv(0x12345678)
    test_pokerng_mod = PokeRNGMod(0x12345678)
    test_arng = ARNG(0x12345678)
    test_xdrng = XDRNG(0x12345678)

    test_pokerngr_div = PokeRNGRDiv(0x12345678)
    test_pokerngr_mod = PokeRNGRMod(0x12345678)
    test_arngr = ARNGR(0x12345678)
    test_xdrngr = XDRNGR(0x12345678)
    assert tuple(
        tuple(test_pokerng_div.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 1, 1, 0),
        (2, 2, 0, 2, 1),
        (19, 1, 11, 14, 3),
        (70, 74, 25, 72, 68),
        (230, 42, 92, 133, 221),
    )
    assert tuple(
        tuple(test_pokerng_mod.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 0, 0, 0, 0),
        (2, 2, 3, 3, 3),
        (8, 3, 18, 2, 0),
        (94, 1, 8, 41, 61),
        (76, 93, 38, 12, 132),
    )
    assert tuple(
        tuple(test_arng.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 0, 0, 1),
        (2, 4, 3, 4, 1),
        (1, 13, 11, 3, 20),
        (15, 33, 57, 36, 87),
        (105, 49, 75, 45, 63),
    )
    assert tuple(
        tuple(test_xdrng.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 1, 1, 1, 0),
        (2, 3, 3, 3, 2),
        (3, 17, 4, 0, 12),
        (35, 44, 94, 7, 23),
        (8, 0, 173, 89, 232),
    )

    assert tuple(
        tuple(test_pokerngr_div.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 0, 1, 1),
        (4, 2, 3, 1, 4),
        (3, 10, 15, 24, 15),
        (86, 91, 20, 45, 44),
        (143, 22, 59, 75, 170),
    )
    assert tuple(
        tuple(test_pokerngr_mod.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 1, 0, 1, 0),
        (3, 1, 0, 3, 4),
        (19, 4, 9, 18, 2),
        (56, 39, 23, 86, 53),
        (51, 229, 5, 237, 186),
    )
    assert tuple(
        tuple(test_arngr.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 1, 1, 1),
        (4, 1, 1, 0, 0),
        (6, 22, 15, 11, 1),
        (18, 95, 43, 14, 43),
        (65, 82, 251, 148, 64),
    )
    assert tuple(
        tuple(test_xdrngr.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 1, 0, 1),
        (3, 4, 4, 0, 1),
        (19, 3, 21, 8, 23),
        (14, 84, 97, 77, 67),
        (253, 125, 122, 133, 99),
    )
