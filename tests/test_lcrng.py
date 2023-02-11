"""Tests for LCRNG classes"""
from numba_pokemon_prngs.lcrng import PokeRNGDiv, PokeRNGMod


def test_lcrng_next():
    """Test LCRNG next() calls for full seed"""
    test_lcrng_div = PokeRNGDiv(0x12345678)
    test_lcrng_mod = PokeRNGMod(0x12345678)
    assert tuple(test_lcrng_div.next() for _ in range(5)) == (
        192004491,
        2229936802,
        3649731437,
        4108329948,
        646229279,
    )
    assert tuple(test_lcrng_mod.next() for _ in range(5)) == (
        192004491,
        2229936802,
        3649731437,
        4108329948,
        646229279,
    )


def test_lcrng_next_u16():
    """Test LCRNG next_u16() calls for 16-bit rand"""
    test_lcrng_div = PokeRNGDiv(0x12345678)
    test_lcrng_mod = PokeRNGMod(0x12345678)
    assert tuple(test_lcrng_div.next_u16() for _ in range(5)) == (
        2929,
        34026,
        55690,
        62688,
        9860,
    )
    assert tuple(test_lcrng_mod.next_u16() for _ in range(5)) == (
        2929,
        34026,
        55690,
        62688,
        9860,
    )


def test_lcrng_next_rand():
    """Test LCRNG next_rand() calls for constrained rand"""
    test_lcrng_div = PokeRNGDiv(0x12345678)
    test_lcrng_mod = PokeRNGMod(0x12345678)
    assert tuple(
        tuple(test_lcrng_div.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 1, 1, 1, 0),
        (2, 2, 0, 2, 1),
        (19, 1, 11, 14, 3),
        (70, 74, 25, 72, 68),
        (230, 42, 92, 133, 221),
    )
    assert tuple(
        tuple(test_lcrng_mod.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 0, 0, 0, 0),
        (2, 2, 3, 3, 3),
        (8, 3, 18, 2, 0),
        (94, 1, 8, 41, 61),
        (76, 93, 38, 12, 132),
    )
