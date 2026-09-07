# Recipe Provenance Classification: American vs. Italian Cuisine

## Overview
Binary classification task: predict whether a recipe is American or Italian from 40 pre-computed TF-IDF ingredient features. 

Before comparing models, the first step was checking the data against its own documentation as the provided TF-IDF values didn't match the formula stated in the problem description, which had to be resolved before any model comparison would be meaningful.

## Approach/Methods
- Found that the provided TF-IDF values don't match the plain-log IDF formula stated in the problem description; derived and verified the actual weighting used (a log1p-smoothed variant) to machine precision (max residual < 1e-13)
- Found 54% train/test row-level duplication; designed a grouped cross-validation protocol evaluating only on non-duplicated rows to get an honest generalization estimate
- Benchmarked 13 classifier families (logistic regression, KNN, SVM, gradient boosting variants, random forest, Extra-Trees) under that protocol
- Engineered 3 aggregate features (ingredient count, total mass, mean intensity) on top of the 40 raw TF-IDF columns
- Final model: multi-seed Extra-Trees ensemble + exact-pattern lookup for duplicate-matched rows
- Interpreted the model with Gini importance and SHAP (TreeSHAP)

## Key Results
- ~12.8% misclassification rate on novel (non-duplicate) recipes - best of 13 model families benchmarked
- Reconciled the TF-IDF feature values with their actual underlying formula (distinct from the one stated in the problem description), which left unaddressed would have misrepresented ingredient rarity across the entire feature set
- Uncovered 54% train/test row-level duplication; the grouped CV protocol prevents this from inflating reported accuracy


## Tools Used
Python (pandas, numpy, scikit-learn, XGBoost, LightGBM, SHAP)

## Repository Contents
- `n00_eda_tfidf_diagnosis.ipynb`: TF-IDF formula diagnosis and exploratory data analysis
- `n01_model_benchmark.py`: grouped cross-validation benchmark across 13 classifier families
- `n02_final_model.py`: final multi-seed Extra-Trees + exact-pattern lookup pipeline
- `n03_feature_importance.py`: Gini importance and SHAP (TreeSHAP) analysis of the final model
- `report.pdf`: full write-up (methodology, validation design, results, limitations)
- `report/report.tex`: LaTeX source for the report
- `data_dictionary.md`: column descriptions for the 40 ingredient features and the target label
  The dataset (`train.csv`, `test.csv`) was provided by the university for coursework and is not included in this repository. Code is shared to demonstrate methodology
