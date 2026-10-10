# CreditLens: Probability of Default & Portfolio Stress Model

CreditLens is a reproducible credit-risk research project trained on 30,000 real account records from the UCI Default of Credit Card Clients dataset. It answers a practical question: **which accounts are most likely to default next month, and how does a deterioration in repayment behaviour affect portfolio expected loss?**

This is a model, not a calculator or a dashboard. It learns from data, is evaluated on borrowers it did not see during training, produces calibrated default probabilities, and can score new CSV records.

## What it demonstrates

- Financial feature engineering: utilization, repayment capacity, and delinquency persistence
- A transparent logistic scorecard versus nonlinear gradient boosting
- Five-fold cross-validation and a held-out test set
- Probability calibration, ROC-AUC, PR-AUC, and Brier score
- A cost-sensitive manual-review threshold
- Expected credit loss: `PD × LGD × exposure`
- Portfolio stress testing
- Permutation-based model explanations
- A protected-attribute audit; sex is withheld from training and used only for evaluation

## Reproduce the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python -m unittest discover -s tests
```

The first training run downloads the CC BY 4.0 dataset through UCI's official Python client and caches it under `data/`.

## Score new records

```bash
python src/predict.py path/to/uci_format_records.csv --output scored.csv
```

The input must contain raw columns `X1` through `X23`, following the UCI data dictionary. Output contains a calibrated probability of default and a cost-sensitive review flag.

## Project structure

```text
├── artifacts/              # trained model, metrics, predictions, audit
├── data/                   # cached source dataset
├── figures/                # publication-ready evaluation charts
├── src/features.py         # feature engineering and stress scenario
├── src/train.py            # training, selection, calibration, evaluation
├── src/predict.py          # batch inference CLI
├── tests/                  # financial and transformation tests
├── MODEL_CARD.md           # governance, limitations, intended use
└── REPORT.md               # research-style findings
```

## Data

Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. DOI: 10.24432/C55S3H. The dataset contains 30,000 Taiwanese credit-card clients, 23 explanatory variables, and a next-month default outcome.

## Important limitation

This is an educational research model, not a production underwriting system. The records are from Taiwan in 2005, the stress is hypothetical, and the model has not been validated for current lending decisions, other populations, or legal compliance.
