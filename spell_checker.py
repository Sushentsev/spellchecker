from typing import List, Optional

from spylls.hunspell import Dictionary

from constructors import ranker_ctor, build_features
from dataset import DatasetEntry
from features.base import Feature


class SpellChecker:
    def __init__(self, dictionary: Dictionary, suggestions: int,
                 features: Optional[List[Feature]] = None, ranker_type: str = "basic"):
        self._dictionary = dictionary
        self._suggestions = suggestions
        self._candidates_ranker = ranker_ctor(features or [], ranker_type)

    def train(self, train_dataset: List[DatasetEntry],
              val_dataset: Optional[List[DatasetEntry]] = None) -> "SpellChecker":
        self._candidates_ranker.fit(train_dataset, val_dataset)
        return self

    def ranked_candidates(self, word: str) -> List[str]:
        candidates = list(self._dictionary.suggester.ngram_suggestions(word, set()))

        if len(candidates) == 0:
            return []

        ranked_candidates = self._candidates_ranker.rank(word, candidates)
        return ranked_candidates[:self._suggestions]

    def rectify_word(self, word: str) -> str:
        if self._dictionary.lookup(word):
            return word

        candidates = self.ranked_candidates(word)
        return candidates[0] if len(candidates) > 0 else word

    @staticmethod
    def from_files(dict_files_path: str, features_files_path: str,
                   suggestions: int = 10, ranker_type: str = "basic") -> "SpellChecker":
        dictionary = Dictionary.from_files(dict_files_path)
        features = build_features(features_files_path)
        return SpellChecker(dictionary, suggestions, features, ranker_type)
