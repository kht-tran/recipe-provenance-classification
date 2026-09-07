"""
A grouped cross-validation benchmark of
13 classifier families on the 40 raw TF-IDF ingredient features, plus a
43-feature Extra-Trees variant (the final model's feature set).

Validation protocol (singleton/duplicate grouped CV):
Training rows are keyed by their 40-feature tuple. Rows whose tuple is
duplicated elsewhere in train.csv are always kept in the training fold.
Only "singleton" rows (a unique 40-tuple) rotate through 5 repetitions of
stratified 3-fold CV as the held-out test set, so every test row is
provably novel (no identical row was ever seen during training) --
giving 15 error estimates per model.

Input:  train.csv (40 TF-IDF feature columns + `y` label column)
Output: printed summary table (mean/std error, acc) and
        tab_model_comparison.tex (LaTeX table for the report)
"""
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

warnings.filterwarnings("ignore")

N_FOLDS = 3
N_REPEATS = 5
BASE_SEED = 300

tr = pd.read_csv("train.csv")
feat = [c for c in tr.columns if c != "y"]
Xraw = tr[feat].values
y = (tr["y"].values == 2).astype(int)
keys = tr[feat].apply(tuple, axis=1).values
cnt = Counter(keys)

is_sing = np.array([cnt[k] == 1 for k in keys])
sing_idx = np.where(is_sing)[0]
dup_idx  = np.where(~is_sing)[0]
print(f"singletons={len(sing_idx)}, dup-rows={len(dup_idx)}", flush=True)

# engineered 43-feature matrix
n_ing   = (Xraw > 0).sum(1, keepdims=True).astype(float)
tot     = Xraw.sum(1, keepdims=True)
mean_int = np.where(n_ing > 0, tot / n_ing, 0.0)
Xeng    = np.hstack([Xraw, n_ing, tot, mean_int])

MODELS = {
    "LDA":             (LinearDiscriminantAnalysis(), Xraw),
    "BernoulliNB":     (BernoulliNB(), Xraw),
    "ComplementNB":    (ComplementNB(), Xraw),
    "LogReg-L2":       (make_pipeline(StandardScaler(),
                                       LogisticRegression(penalty="l2",
                                           C=1.0, max_iter=2000)),
                        Xraw),
    "LogReg-L1":       (make_pipeline(StandardScaler(),
                                       LogisticRegression(penalty="l1",
                                           solver="liblinear", C=1.0,
                                           max_iter=2000, random_state=42)),
                        Xraw),
    "LogReg-EN":       (make_pipeline(StandardScaler(),
                                       LogisticRegression(penalty="elasticnet",
                                           solver="saga", l1_ratio=0.5,
                                           C=1.0, max_iter=5000)),
                        Xraw),
    "KNN-k5-cosine":   (KNeighborsClassifier(n_neighbors=5, metric="cosine"),
                        Xraw),
    "SVM-RBF":         (make_pipeline(StandardScaler(),
                                       SVC(C=4.0, gamma=0.05,
                                           probability=False,
                                           random_state=42)),
                        Xraw),
    "HistGBM":         (HistGradientBoostingClassifier(max_iter=300,
                                                        random_state=42),
                        Xraw),
    "RandomForest":    (RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                               random_state=42),
                        Xraw),
    "ExtraTrees-40":   (ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                              random_state=42),
                        Xraw),
    "ExtraTrees-43":   (ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                              random_state=42),
                        Xeng),
}
if _HAS_XGB:
    MODELS["XGBoost"] = (XGBClassifier(n_estimators=300, learning_rate=0.05,
                                        max_depth=6, subsample=0.8,
                                        eval_metric="logloss", n_jobs=-1,
                                        random_state=42, verbosity=0), Xraw)
if _HAS_LGB:
    MODELS["LightGBM"] = (LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                          num_leaves=31, subsample=0.8,
                                          verbose=-1, random_state=42), Xraw)

res = {nm: [] for nm in MODELS}
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=BASE_SEED + rep)
    for fold, (s_tr, s_te) in enumerate(skf.split(sing_idx, y[sing_idx])):
        tr_rows = np.concatenate([dup_idx, sing_idx[s_tr]])
        te_rows = sing_idx[s_te]
        ytr, yte = y[tr_rows], y[te_rows]

        for nm, (model, X) in MODELS.items():
            m = clone(model).fit(X[tr_rows], ytr)
            err = float((m.predict(X[te_rows]) != yte).mean())
            res[nm].append(err)

        print(f"rep{rep} fold{fold} n_test={len(te_rows)}", flush=True)

summary = pd.DataFrame(
    {nm: (np.mean(vals), 1 - np.mean(vals), np.std(vals))
     for nm, vals in res.items()},
    index=["err%", "acc%", "std%"],
).T.mul(100).sort_values("err%")

print("\n=== Novel-row error (singleton 3-fold x 5 reps) ===")
print(summary.round(2))

# --- LaTeX table output (display order groups by family) ---
# Display order: groups worst-to-best, within each group worst-to-best.
# Built dynamically so the ordering reflects actual results.
def make_display(stats):
    boosting = [k for k in [("XGBoost",  r"XGBoost"),
                             ("LightGBM", r"LightGBM"),
                             ("HistGBM",  r"HistGradientBoosting")]
                if k[0] in stats]
    groups = [
        [("LDA",          r"LDA"),
         ("ComplementNB", r"Complement Naive Bayes"),
         ("BernoulliNB",  r"Bernoulli Naive Bayes")],
        [("LogReg-L2",  r"Logistic regression (L2)"),
         ("LogReg-EN",  r"Logistic regression (elastic net)"),
         ("LogReg-L1",  r"Logistic regression (L1)")],
        [("KNN-k5-cosine",  r"KNN (cosine, $k{=}5$)")],
        boosting,
        [("SVM-RBF",    r"SVM (RBF kernel)")],
        [("RandomForest",   r"Random forest"),
         ("ExtraTrees-40",  r"Extra-Trees (40 features)"),
         ("ExtraTrees-43",  r"Extra-Trees (43 features, final)")],
    ]
    # drop empty groups (e.g. boosting if neither xgb nor lgb installed)
    groups = [g for g in groups if g]
    # sort within each group by error descending (worst first)
    for g in groups:
        g.sort(key=lambda x: -stats[x[0]][0])
    # sort groups by their worst (highest-error) member, worst group first
    groups.sort(key=lambda g: max(stats[k][0] for k, _ in g), reverse=True)

    display = []
    for i, g in enumerate(groups):
        for j, (key, label) in enumerate(g):
            display.append((key, label, (i > 0 and j == 0)))
    return display

stats = {}
for nm, vals in res.items():
    arr = np.array(vals)
    stats[nm] = (arr.mean(), arr.std())

DISPLAY = make_display(stats)

lines = []
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\toprule")
lines.append(r"Method & Error\,(\%) & Std\,(\%) \\")
lines.append(r"\midrule")
for key, label, sep in DISPLAY:
    if sep:
        lines.append(r"\addlinespace[4pt]")
    mu, sd = stats[key]
    lines.append(rf"{label} & ${mu*100:.2f}$ & ${sd*100:.2f}$ \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

tex = "\n".join(lines)
with open("tab_model_comparison.tex", "w") as f:
    f.write(tex + "\n")
print("\nWrote tab_model_comparison.tex")
print(tex)
