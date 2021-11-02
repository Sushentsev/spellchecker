from typing import List, Iterator, Optional, Tuple

import numpy as np
from catboost import CatBoostRanker, Pool
from tqdm import tqdm

from dataset import DatasetEntry
from features.base import Feature
from scorers.base import CandidatesRanker


def convert(dataset: List[DatasetEntry]) -> Tuple[List[str], List[List[str]], List[str]]:
    queries, docs, labels = [], [], []
    for entry in dataset:
        queries.append(entry.misspelled)
        labels.append(entry.correct)
        docs.append(entry.candidates)

    return queries, docs, labels


class BoostingRanker(CandidatesRanker):
    def __init__(self, features: List[Feature]):
        super(BoostingRanker, self).__init__(features)
        self._ranker = CatBoostRanker(verbose=False, loss_function="YetiRankPairwise")

    def _create_pool(self, queries: List[str], docs: List[List[str]], labels: Optional[List[str]] = None) -> Pool:
        X, queries_pool, labels_pool = [], [], []

        for query_id in tqdm(range(len(queries)), "Creating pool"):
            for doc_id in range(len(docs[query_id])):
                queries_pool.append(query_id + 1)
                features = [feature(queries[query_id], docs[query_id][doc_id]) for feature in self._features]
                X.append(features)

                if labels is not None:
                    labels_pool.append(0)

            if labels is not None:
                queries_pool.append(query_id + 1)
                features = [feature(queries[query_id], labels[query_id]) for feature in self._features]
                X.append(features)

                if labels is not None:
                    labels_pool.append(1)

        if labels is None:
            return Pool(data=X, group_id=queries_pool)

        return Pool(data=X, group_id=queries_pool, label=labels_pool)

    def fit(self, train_dataset: List[DatasetEntry],
            val_dataset: Optional[List[DatasetEntry]] = None) -> "CandidatesRanker":
        train_pool = self._create_pool(*convert(train_dataset))

        if val_dataset is not None:
            self._ranker.fit(train_pool, eval_set=self._create_pool(*convert(val_dataset)))
        else:
            self._ranker.fit(train_pool)

        return self

    def rank(self, misspelled: str, candidates: Iterator[str]) -> List[str]:
        candidates = list(candidates)

        if len(candidates) == 0:
            return []

        pool = self._create_pool([misspelled], [candidates])
        scores = self._ranker.predict(pool)
        return list(np.array(candidates)[np.argsort(-scores)])
