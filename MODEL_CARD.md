# Model Card — CreditLens

## Intended use

Educational analysis of next-month default risk and portfolio stress. The model can rank records for hypothetical manual review and demonstrate how expected loss changes under altered repayment behaviour.

## Not intended for

- Real lending, credit approval, pricing, or limit decisions
- Use outside the source population without revalidation
- Automated adverse action
- Claims about causality

## Data

The UCI Default of Credit Card Clients dataset contains 30,000 records from Taiwan, with payment history from April–September 2005 and a next-month default label. It has no missing values. The historical and geographic context creates substantial transportability risk.

## Target

`default = 1` indicates default in the following month.

## Protected attributes

Sex is deliberately excluded from the model matrix. It is retained only for a post-model diagnostic comparing actual default rate, average predicted probability, and review rate. This audit is descriptive and is not proof of fairness.

## Validation

- Stratified 75/25 train-test split with a fixed seed
- Five-fold cross-validation inside the training data
- Held-out ROC-AUC, PR-AUC, and Brier score
- Probability calibration using sigmoid calibration
- Directional stress test

## Known limitations

- The data are historical and may encode past policy or social bias.
- A random split does not test temporal drift.
- Credit limits are used as an exposure proxy; true EAD is unavailable.
- The assumed 60% loss-given-default is illustrative.
- The repayment shock is a scenario, not a forecast.
- Group averages can hide intersectional disparities.

## Governance recommendation

Any real use would require current representative data, temporal validation, drift monitoring, jurisdiction-specific fair-lending review, adverse-action reason testing, and independent model-risk validation.
