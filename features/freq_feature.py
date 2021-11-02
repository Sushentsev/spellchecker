import os
from typing import Dict

import numpy as np
import pandas as pd

from features.base import Feature


class FreqFeature(Feature):
    def __init__(self, freqs: Dict[str, float]):
        self._freqs = freqs
        self._default_freq = np.mean(list(self._freqs.values()))

    def __call__(self, misspelled: str, candidate: str) -> float:
        return self._freqs.get(candidate, self._default_freq)

    @staticmethod
    def from_files(path: str) -> "FreqFeature":
        freqs_df = pd.read_csv(os.path.join(path, "freqs.csv")).set_index("word")
        freqs = freqs_df.to_dict()["count"]
        return FreqFeature(freqs)
