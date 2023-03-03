"""Namespace for EncounterArea3 (..data.encounter.encounter_area_3) related functions"""

import numpy as np
from ..data.encounter.encounter_area_3 import EncounterArea3, Slot3
from ..lcrng import PokeRNGMod
from ..compilation import optional_njit
from ..enums import Encounter


def compute_table(ranges: tuple[int], greater: bool = False) -> np.array:
    """Compute encounter slot lookup table"""
    table = np.empty(100, dtype=np.uint8)

    r_val = 99 if greater else 0
    for i, range_val in enumerate(ranges):
        for r_val in range(r_val, range_val, -1 if greater else 1):
            table[r_val] = i

    return table


GRASS_TABLE = compute_table((20, 40, 50, 60, 70, 80, 85, 90, 94, 98, 99, 100))
OLD_ROD_TABLE = compute_table((70, 100))
GOOD_ROD_TABLE = compute_table((60, 80, 100))
SUPER_ROD_TABLE = compute_table((40, 80, 95, 99, 100))
ROCK_SMASH_AND_SURF_TABLE = compute_table((60, 90, 95, 99, 100))

hslot_rand = PokeRNGMod.const_rand(100)


@optional_njit
def calculate_hslot(
    encounter_area: EncounterArea3, rng: PokeRNGMod, encounter_type: Encounter
) -> Slot3:
    """Calculate the Slot3 of the current rng"""
    if encounter_type == Encounter.OLD_ROD:
        return encounter_area.fish_old[OLD_ROD_TABLE[hslot_rand(rng)]]
    if encounter_type == Encounter.GOOD_ROD:
        return encounter_area.fish_good[GOOD_ROD_TABLE[hslot_rand(rng)]]
    if encounter_type == Encounter.SUPER_ROD:
        return encounter_area.fish_super[SUPER_ROD_TABLE[hslot_rand(rng)]]
    if encounter_type == Encounter.ROCK_SMASH:
        return encounter_area.rock[ROCK_SMASH_AND_SURF_TABLE[hslot_rand(rng)]]
    if encounter_type == Encounter.SURFING:
        return encounter_area.water[ROCK_SMASH_AND_SURF_TABLE[hslot_rand(rng)]]
    return encounter_area.land[GRASS_TABLE[hslot_rand(rng)]]
