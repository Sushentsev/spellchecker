from abc import ABC, abstractmethod
from typing import List, Optional

from dataset import DatasetEntry
from features.base import Feature


class CandidatesRanker(ABC):
    def __init__(self, features: List[Feature]):
        self._features = features

    @abstractmethod
    def fit(self, train_dataset: List[DatasetEntry],
            val_dataset: Optional[List[DatasetEntry]] = None) -> "CandidatesRanker":
        raise NotImplementedError

    def rank(self, misspelled: str, candidates: List[str]) -> List[str]:
        raise NotImplementedError
