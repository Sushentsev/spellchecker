from pyphonetics import RefinedSoundex

from features.base import Feature


class PhoneticDistFeature(Feature):
    def __init__(self):
        self._rs = RefinedSoundex()

    def __call__(self, misspelled: str, candidate: str) -> float:
        return -self._rs.distance(misspelled, candidate, metric="levenshtein")


    @staticmethod
    def from_files(path: str) -> "PhoneticDistFeature":
        return PhoneticDistFeature()
