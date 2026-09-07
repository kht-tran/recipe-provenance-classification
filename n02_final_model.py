from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier

N_SEEDS = 30
ET_PARAMS = dict(n_estimators=600, max_features="sqrt", 
                 min_samples_leaf=1, max_depth=None)

def add_features(df, base_cols):
    out = df.copy()
    base = df[base_cols]
    out["n_ingredients"] = (base > 0).sum(axis=1)
    out["total_mass"] = base.sum(axis=1)
    out["mean_intensity"] = out["total_mass"] / (out["n_ingredients"] + 1e-9)
    return out

class MultiSeedET(ClassifierMixin, BaseEstimator):
    def __init__(self, params, n_seeds):
        self.params = params
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = [
            ExtraTreesClassifier(**self.params, random_state=s, n_jobs=-1).fit(X, y)
            for s in range(self.n_seeds)]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def main():
    print("Loading data")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    base_cols = [c for c in train.columns if c != "y"]

    Xtr_raw = train[base_cols]
    ytr = train["y"].map({1: 0, 2: 1}).values
    Xte_raw = test[base_cols]
    print(f"train={len(train)}  test={len(test)}  features={len(base_cols)}")

    Xtr = add_features(Xtr_raw, base_cols).values
    Xte = add_features(Xte_raw, base_cols).values

    print("Building exact-pattern lookup")
    keys_tr = [tuple(r) for r in Xtr_raw.values]
    keys_te = [tuple(r) for r in Xte_raw.values]
    buckets = defaultdict(list)
    for k, lab in zip(keys_tr, ytr):
        buckets[k].append(lab)
    lookup = {k: Counter(v).most_common(1)[0][0] for k, v in buckets.items()}

    print(f"Fitting {N_SEEDS}-seed ET")
    model = MultiSeedET(ET_PARAMS, N_SEEDS).fit(Xtr, ytr)

    print("Predicting")
    model_pred = model.predict(Xte)
    final_pred = model_pred.copy()
    for j, k in enumerate(keys_te):
        if k in lookup:
            final_pred[j] = lookup[k]

    out = "predictions_et30_lookup.txt"
    pd.Series(final_pred).map({0: 1, 1: 2}).to_csv(out, index=False, header=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
