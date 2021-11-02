# SpellChecker
Homework 6 from "Production stories" HSE SPb course.

## Solution

To extract candidates, n-grams index is used. 
After that, 6 features are extracted for each candidate.
Ranking is done in two ways: 
- ``Basic`` ranking — scaler and average across all features
- ``Boosing`` ranking — ranking with ``CatBoost`` library 
  and ``YetiRankPairwise`` loss function

## Installation

To install packages use: 
```angular2html
pip install -r requirements.txt
```

## Usage

To train and evaluate model use:
```
python main.py \
 --data_path=<data_path> \
 --ranker_type=<ranker_type>
```

where 
- ``<data_path>`` — directory with train, test and features data
- ``<ranker_type>`` — type of ranking (``basic`` or ``boosting``)

## Scores

[Test data.](http://aspell.net/test/cur/)

| | Basic | Boosting |
| ------| ----------- | --- |
| Acc@1 | 0.52 | 0.59 |
| Acc@3 | 0.70 | 0.72 |
| Acc@10 | 0.72 | 0.73 |
| MRR | 0.60 | 0.65 |
