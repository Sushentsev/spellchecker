from typing import List

from features.base import Feature
from features.freq_feature import FreqFeature
from features.edit_dist_feature import EditDistFeature
from features.keyboard_dist_feature import KeyboardDistFeature
from features.length_dist_feature import LengthDistFeature
from features.n_gram_overlap_feature import NGramOverlapFeature
from features.phonetic_dist_feature import PhoneticDistFeature
from rankers.base import CandidatesRanker
from rankers.basic_ranker import BasicRanker
from rankers.boosting_ranker import BoostingRanker

features_list = [
    EditDistFeature, FreqFeature, KeyboardDistFeature, LengthDistFeature,
    NGramOverlapFeature, PhoneticDistFeature,
]


def build_features(files_path: str) -> List[Feature]:
    return [feature.from_files(files_path) for feature in features_list]


def ranker_ctor(features: List[Feature], scorer_type: str) -> CandidatesRanker:
    if scorer_type == "basic":
        return BasicRanker(features)
    elif scorer_type == "boosting":
        return BoostingRanker(features)
    else:
        raise ValueError("Invalid ranker type.")
