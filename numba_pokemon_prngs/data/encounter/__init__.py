"""Encounter data for Pokemon games"""

import re
import json
import importlib.resources as pkg_resources
import numpy as np
from .encounter_area_3 import EncounterArea3
from .. import CONSTANT_CASE_SPECIES_EN
from ...resources import encount


def clean_map_string(map_string: str) -> str:
    """Clean a map string to be easily readable"""
    map_string = map_string.replace("MAP_", "").replace("_", " ")

    strings = map_string.split()
    for i, string in enumerate(strings):
        if match := re.match(r"Route(\d+)", string, re.IGNORECASE):
            strings[i] = f"Route {match[1]}"
        elif match := re.match(r"Room(\d+)", string, re.IGNORECASE):
            strings[i] = f"Room {match[1]}"
        elif re.match(r"B(\d+)F", string):
            strings[i] = string
        elif re.match(r"(\d+)F", string):
            strings[i] = string
        elif re.match(r"(\d+)R", string):
            strings[i] = string
        else:
            strings[i] = string.capitalize()

    return " ".join(strings)


def load_e_encounter() -> tuple[tuple[str], np.recarray]:
    """Load emerald encounter data"""
    with pkg_resources.open_text(
        encount, "e_wild_encounters.json"
    ) as e_wild_encounters:
        encounters = json.load(e_wild_encounters)["wild_encounter_groups"][0][
            "encounters"
        ]
    encounter_information = np.zeros(len(encounters), dtype=EncounterArea3.dtype).view(
        np.recarray
    )
    map_names = []
    for map_number, encounter in enumerate(encounters):

        if re.match(r"gAlteringCave[2-9]", encounter["base_label"]):
            map_names.append("")
            continue

        if "Unused" in encounter["base_label"]:
            map_names.append("")
            continue

        map_names.append(clean_map_string(encounter["map"]))

        encounter_area: EncounterArea3 = encounter_information[map_number]

        land_mons = "land_mons" in encounter
        water_mons = "water_mons" in encounter
        rock_smash_mons = "rock_smash_mons" in encounter
        fishing_mons = "fishing_mons" in encounter

        if land_mons:
            encounter_area.land_rate = encounter["land_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["land_mons"]["mons"]):
                encounter_slot = encounter_area.land[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
        if water_mons:
            encounter_area.water_rate = encounter["water_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["water_mons"]["mons"]):
                encounter_slot = encounter_area.water[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
        if rock_smash_mons:
            encounter_area.rock_rate = encounter["rock_smash_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["rock_smash_mons"]["mons"]):
                encounter_slot = encounter_area.rock[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
        if fishing_mons:
            encounter_area.fish_rate = encounter["fishing_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["fishing_mons"]["mons"]):
                encounter_slot = encounter_area.fish[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])

    return tuple(map_names), encounter_information


MAP_NAMES_E, ENCOUNTER_INFORMATION_E = load_e_encounter()
