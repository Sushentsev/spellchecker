from abc import ABC, abstractmethod


class Feature(ABC):
    @abstractmethod
    def __call__(self, misspelled: str, candidate: str) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_files(path: str) -> "Feature":
        raise NotImplementedError
