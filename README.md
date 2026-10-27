# Credit Scoring Model

Predicts the likelihood that a borrower will become seriously delinquent
(90+ days past due) within two years, using past financial data.
Built for the CodeAlpha Machine Learning Internship (Task 1).

## Dataset
Kaggle "Give Me Some Credit" competition, `cs-training.csv`
(150,000 borrowers, 11 features). The data is not included in this repo.
Download it from the competition's Data tab and place it in a `data/` folder.

## What I did
1. **Explored the data:** 6.7% of borrowers defaulted (heavy class imbalance).
2. **Cleaned it:** removed an invalid age of 0, found that 96/98 in the
   late-payment columns were codes and not real counts, filled missing income
   and dependents with training-set medians, and capped extreme values at the
   99th percentile.
3. **Engineered features:** `LateCodeFlag` (the coded rows defaulted 55% of the
   time against 6.6% for everyone else) and `IncomeMissing`.
4. **Split before preprocessing:** 80/20 stratified split, with all medians and
   caps learned from the training set only, to avoid data leakage.
5. **Trained two models:** Logistic Regression and Random Forest, both with
   balanced class weights.

## Results (test set, 30,000 borrowers)
| Model | Recall (defaulters) | Precision (defaulters) | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.75 | 0.22 | 0.34 | 0.862 |
| Random Forest | 0.79 | 0.21 | 0.34 | 0.867 |

## Key findings
- Credit utilization and late-payment history are the strongest predictors;
  income and debt ratio matter much less.
- The two models perform almost the same, so the limit is the information in
  the data, not the algorithm.
- Accuracy is misleading here: a model that predicts "no default" for everyone
  scores 93%. Recall, precision, F1 and ROC-AUC are the metrics that matter.
- There is a trade-off: catching more defaulters means wrongly flagging more
  good customers. The right cutoff depends on a lender's costs.

## Limitations
Scores are rankings, not true probabilities, because of class balancing. This
is a learning project and should not be used for real lending decisions.

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Then open `credit_scoring_model.ipynb` and run all cells.