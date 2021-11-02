from typing import Set

from features.base import Feature


class NGramOverlapFeature(Feature):
    def __init__(self):
        self._n = 2

    def _ngrams(self, word: str) -> Set[str]:
        word = f"${word}$"
        return {word[i:(i + self._n)] for i in range(0, len(word) - self._n + 1)}

    def __call__(self, misspelled: str, candidate: str) -> float:
        return len(self._ngrams(misspelled) & self._ngrams(candidate))

    @staticmethod
    def from_files(path: str) -> "NGramOverlapFeature":
        return NGramOverlapFeature()
