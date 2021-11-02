import numpy as np

from features.base import Feature


class LengthDistFeature(Feature):
    def __call__(self, misspelled: str, candidate: str) -> float:
        return -np.abs(len(misspelled) - len(candidate))

    @staticmethod
    def from_files(path: str) -> "Feature":
        return LengthDistFeature()
