"""Tests for Mersenne Twister classes"""
from hashlib import sha256
from numba_pokemon_prngs.mersenne_twister import (
    MersenneTwister,
    SIMDFastMersenneTwister,
    TinyMersenneTwister,
)


def test_init():
    """Test initialization re_init() functions"""
    test_mt = MersenneTwister(0)
    test_sfmt = SIMDFastMersenneTwister(0)
    test_tinymt = TinyMersenneTwister(0)

    assert tuple(
        sha256(test_mt.state.data.tobytes()).hexdigest()
        for _ in (
            test_mt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "f75f425d88ef519f1b972ccc688c2659e699934204c3af00f8527d5aaec5db81",
        "abfdf49e18c3e57a2ae366c11f95252082872085e5048a8d475751bd09bea408",
        "af5ae4b6d7dce444709d7fe97f718c698f2d7076fcc8a62384a1fa2ee2246f17",
        "006d3165256376db0287398b900c8eb38828f2a23e298c93167597d61fca3729",
    )

    assert tuple(
        sha256(test_sfmt.state.tobytes()).hexdigest()
        for _ in (
            test_sfmt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "ccc11319e23c70d1a94dd6d3fefa1086921674e59e72c531c2cc27544320b21d",
        "abfdf49e18c3e57a2ae366c11f95252082872085e5048a8d475751bd09bea408",
        "af5ae4b6d7dce444709d7fe97f718c698f2d7076fcc8a62384a1fa2ee2246f17",
        "006d3165256376db0287398b900c8eb38828f2a23e298c93167597d61fca3729",
    )

    assert tuple(
        sha256(test_tinymt.state.tobytes()).hexdigest()
        for _ in (
            test_tinymt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "3638ef812de23a06ffbec0c4e9abc3c155cf0912c3385e14d7173cb5d3937a39",
        "60b6d3900e32a93c9966652ecfbc8fb4cd72a20c2587ec8bcb62427be3068bb5",
        "953690085212afe1596413f24da480062d82b4d36f9dafc219ae321ca42ce658",
        "6676b3aa5a324abc261c3b406abb192c1c44da1ae88b99f2f1100cb5b597f7c8",
    )


def test_shuffle():
    """Test state transition shuffle() functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)

    assert tuple(
        sha256(test_mt.state.data.tobytes()).hexdigest()
        for _ in (test_mt.shuffle() for _ in range(4))
    ) == (
        "324aa1b936d575034bdac4a7496fff66ad97a712ff8245ef1e133556d214644d",
        "61512e29c9a22c48af8bd959f630af79e869829d6a9e2387e74ea44b2971ef87",
        "c88eaf942ad61179f1b8f11980ec989ab73f51c1d6d4e12494d1f1a37530dd02",
        "60ef09a8fbb7b2cc42c4ce34cb824ffedb4ca5495eb665d58cef5affbf191f22",
    )

    assert tuple(
        sha256(test_sfmt.state.tobytes()).hexdigest()
        for _ in (test_sfmt.shuffle() for _ in range(4))
    ) == (
        "d9ee5bdb83f403eb986e5a7631f24cc89b597d5934ef4b203b23657105cb6393",
        "da5879fbf2cd08855f7539d349c8df3c32c223482c28acd80b9e3dbde56b2cb9",
        "d64b5657cccee15f17bf49e1dd901eeb91db137d45ac02d0c8fd0525a060a914",
        "31b8d4454a387b720735f9d0ce6e7de582afdf2c49e1a2f1b5adaccc86d18773",
    )

    assert tuple(
        sha256(test_tinymt.state.tobytes()).hexdigest()
        for _ in (test_tinymt.shuffle() for _ in range(4))
    ) == (
        "50f5074de3eeb415f816948588d407261c0833b1b2bcb0661ac92719a6c7dcc8",
        "23e7114485474485b4ca928c1dbe6bf05dc302951cfcddbbd06a2b8c891a99a5",
        "59b44e0b82d3464ffc8978d100796c153355da25fb45538953e9d04c900b4e7f",
        "633391236055e9a4b8d9a144d834990c96fb919cb30379d508251cc7aba4b5ba",
    )


def test_next():
    """Test state access/tempering of next() functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)
    assert tuple(test_mt.next() for _ in range(5)) == (
        3331822403,
        157471482,
        2805605540,
        3776487808,
        3041352379,
    )
    assert tuple(test_sfmt.next() for _ in range(5)) == (
        4883196442416872734,
        15199411253426940632,
        10268751338100151066,
        602836503723090224,
        2976335783597749142,
    )
    assert tuple(test_tinymt.next() for _ in range(5)) == (
        1025822,
        2123361235,
        2978809828,
        2070750042,
        1879513653,
    )


def test_next_rand():
    """Test next_rand()/next_rand_mod() bounded rand functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)

    assert tuple(
        tuple(test_mt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 0, 1, 1, 1),
        (1, 2, 4, 1, 2),
        (17, 12, 1, 22, 1),
        (54, 22, 61, 83, 69),
        (11, 140, 12, 76, 59),
    )
    assert tuple(
        tuple(test_mt.next_rand_mod(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 1, 1, 0, 0),
        (3, 1, 1, 0, 1),
        (4, 24, 19, 10, 20),
        (41, 18, 71, 28, 88),
        (154, 172, 65, 161, 125),
    )

    # next_rand is next_rand_mod
    assert tuple(
        tuple(test_sfmt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 0, 0, 0, 0),
        (1, 3, 3, 1, 3),
        (1, 12, 10, 23, 10),
        (61, 24, 95, 71, 43),
        (241, 85, 35, 253, 83),
    )

    assert tuple(
        tuple(test_tinymt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1),
        (10, 17, 20, 6, 4),
        (42, 79, 24, 10, 28),
        (60, 77, 117, 42, 145),
    )
    assert tuple(
        tuple(test_tinymt.next_rand_mod(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 0, 0, 0, 0),
        (2, 2, 1, 3, 4),
        (1, 0, 9, 20, 24),
        (76, 76, 89, 3, 13),
        (206, 170, 162, 66, 85),
    )
