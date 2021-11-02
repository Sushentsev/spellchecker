from dataclasses import dataclass
from typing import List


@dataclass
class DatasetEntry:
    misspelled: str
    correct: str
    candidates: List[str]
