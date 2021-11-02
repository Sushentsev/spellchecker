from features.base import Feature
from textdistance import needleman_wunsch


class KeyboardDistFeature(Feature):
    def __call__(self, misspelled: str, candidate: str) -> float:
        return -needleman_wunsch(misspelled, candidate)

    @staticmethod
    def from_files(path: str) -> "KeyboardDistFeature":
        return KeyboardDistFeature()
