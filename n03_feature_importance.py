import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import shap

from n02_final_model import add_features, ET_PARAMS

SEED = 0

REPORT_GINI_TOP15 = [
    "water", "flour", "total_mass", "mean_intensity", "oil", "white.meat",
    "sugar", "wine", "salt", "garlic", "yeast", "egg", "n_ingredients",
    "fruits", "seeds",
]
REPORT_SHAP_TOP15 = [
    "water", "flour", "wine", "garlic", "yeast", "white.meat", "oil",
    "salt", "sugar", "egg", "seeds", "potato", "total_mass", "onion",
    "n_ingredients",
]


def flag_discrepancies(label, actual_top15, report_top15, k=4):
    actual_topk = set(actual_top15[:k])
    report_topk = set(report_top15[:k])
    dropped = report_topk - actual_topk
    if dropped:
        print(f"[FLAG] {label}: report top-{k} feature(s) {sorted(dropped)} "
              f"not in actual top-{k} {sorted(actual_topk)}")
    else:
        print(f"[ok] {label}: top-{k} agree (order may differ)")


def main():
    train = pd.read_csv("train.csv")
    base_cols = [c for c in train.columns if c != "y"]

    Xtr = add_features(train[base_cols], base_cols)
    feat_names = list(Xtr.columns)
    ytr = train["y"].values
    print(f"train={len(train)}  features={len(feat_names)}")

    et = ExtraTreesClassifier(**ET_PARAMS, random_state=SEED, n_jobs=-1)
    et.fit(Xtr.values, ytr)

    # --- (a) Gini importance ---
    imp = pd.Series(et.feature_importances_, index=feat_names).sort_values(ascending=False)
    top_gini = imp.head(15)

    print("\nTop-15 Gini importance:")
    for rank, (name, val) in enumerate(top_gini.items(), 1):
        print(f"  {rank:2d}. {name:<16s} {val:.4f}")

    plot_order = top_gini.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(range(len(plot_order)), plot_order.values, color="#34495e", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(plot_order)))
    ax.set_yticklabels(plot_order.index, fontsize=9)
    ax.set_xlabel("ET Gini importance")
    ax.set_title("Top-15 feature importance (single-seed, 600 trees)")
    plt.tight_layout()
    plt.savefig("fig_feature_importance.png", dpi=160)
    plt.close()
    print("wrote fig_feature_importance.png")

    # --- (b) TreeSHAP summary (class index for y=2 "Italian") ---
    explainer = shap.TreeExplainer(et)
    sv = explainer.shap_values(Xtr)
    if isinstance(sv, list):
        sv_use = sv[1]
    else:
        sv_arr = np.asarray(sv)
        sv_use = sv_arr[:, :, 1] if sv_arr.ndim == 3 else sv_arr
    print(f"\nSHAP values shape: {np.asarray(sv_use).shape}, X shape: {Xtr.shape}")

    mean_abs_shap = pd.Series(np.abs(sv_use).mean(axis=0), index=feat_names).sort_values(ascending=False)
    top_shap = mean_abs_shap.head(15)

    print("Top-15 mean |SHAP|:")
    for rank, (name, val) in enumerate(top_shap.items(), 1):
        print(f"  {rank:2d}. {name:<16s} {val:.4f}")

    plt.figure(figsize=(7.5, 6.2))
    shap.summary_plot(sv_use, Xtr, show=False, max_display=15, plot_size=None)
    plt.tight_layout()
    plt.savefig("fig_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("wrote fig_shap_summary.png")

    print()
    flag_discrepancies("Gini", list(top_gini.index), REPORT_GINI_TOP15)
    flag_discrepancies("SHAP", list(top_shap.index), REPORT_SHAP_TOP15)


if __name__ == "__main__":
    main()
