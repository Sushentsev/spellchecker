from typing import List, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset import DatasetEntry
from features.base import Feature
from rankers.base import CandidatesRanker


class BasicRanker(CandidatesRanker):
    def __init__(self, features: List[Feature]):
        super(BasicRanker, self).__init__(features)
        self._scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, train_dataset: List[DatasetEntry],
            val_dataset: Optional[List[DatasetEntry]] = None) -> "BasicRanker":
        return self

    def _normalize(self, features: List[List[float]]) -> List[List[float]]:
        if len(features[0]) == 0:
            return features

        return self._scaler.fit_transform(features)

    def rank(self, misspelled: str, candidates: List[str]) -> List[str]:
        features = [[feature(misspelled, candidate) for feature in self._features]
                    for candidate in candidates]

        features = self._normalize(features)
        scores = np.mean(features, axis=1)
        return list(np.array(candidates)[np.argsort(-scores)])
