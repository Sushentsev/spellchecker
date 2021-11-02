from features.base import Feature
from nltk import edit_distance


class EditDistFeature(Feature):
    def __call__(self, misspelled: str, candidate: str) -> float:
        return -edit_distance(misspelled, candidate, transpositions=True)

    @staticmethod
    def from_files(path: str) -> "EditDistFeature":
        return EditDistFeature()
