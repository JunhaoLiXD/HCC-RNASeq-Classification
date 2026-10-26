"""
Sample-level comparison of the predictive models from Step 11.

Each (method, number of genes) pair is one model, giving 2 x 5 = 10 models per
target. A model's label for a sample is the argmax of its out-of-fold class
probabilities averaged over the CV repeats. From the samples x models matrix,
per sample and target:

  - how many models predict each label, and the agreement (share of models
    giving the most common label; 1 = all models agree)
  - the share of models that predict the sample's true label / own cluster
  - the mean probability assigned to the true label across models (a
    sample-level analogue of AUC: 1 = always confidently correct)

Stability of the Tumor / Normal prediction is then compared with stability of
the cluster prediction (k = 2 and k = 3) with three tests per cluster target:
Spearman correlation of agreement, Fisher's exact test on fully stable vs not
(agreement = 1), and Spearman correlation of the mean true-label probability.
All p-values are BH-adjusted together.

Writes sample_by_model.csv, sample_stability.csv, stability_tests.csv,
12_sample_by_model.png and 12_stability.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
META_PATH = DATA_DIR / "metadata_SRP068976.tsv"
FIG_DIR = ROOT / "results" / "assn4" / "figures"
TBL_DIR = ROOT / "results" / "assn4" / "tables"

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}
CORRECT_COLORS = ["#f0b429", "#d9d9d9"]   # wrong, correct

TARGETS = ["group", "kmeans_k2", "kmeans_k3"]
CLUSTER_TARGETS = ["kmeans_k2", "kmeans_k3"]
TARGET_NAMES = {
    "group": "Tumor vs Normal",
    "kmeans_k2": "K-means k = 2",
    "kmeans_k3": "K-means k = 3",
}
METHOD_SHORT = {"logreg": "LR", "rf": "RF"}
LOW_PROB = 0.65         # label samples whose mean true-label probability is below this
ALPHA = 0.05


def load_inputs():
    """Load the Step 11 predictions and short sample labels like 'P39 Tumor'."""
    preds = pd.read_csv(TBL_DIR / "oof_predictions.csv",
                        dtype={"true": str, "pred": str})
    meta = pd.read_csv(META_PATH, sep="\t", index_col=0)
    labels = meta["refinebio_title"].str.replace("Patient ", "P", regex=False)
    return preds, labels


def model_labels(preds: pd.DataFrame, target: str):
    """Per model: repeat-averaged probabilities -> label and true-label prob."""
    sub = preds[preds["target"] == target]
    prob_cols = [c for c in sub.columns
                 if c.startswith("prob_") and sub[c].notna().all()]
    classes = np.array([c[len("prob_"):] for c in prob_cols])

    mean_prob = sub.groupby(["method", "n_genes", "sample"])[prob_cols].mean()
    truth = sub.groupby("sample")["true"].first()

    labels, p_true = {}, {}
    for (method, n_genes), df in mean_prob.groupby(level=[0, 1]):
        df = df.droplevel([0, 1])
        name = f"{METHOD_SHORT[method]}_g{n_genes}"
        labels[name] = pd.Series(classes[df.to_numpy().argmax(axis=1)],
                                 index=df.index)
        true_idx = np.searchsorted(classes, truth.loc[df.index].to_numpy())
        p_true[name] = pd.Series(df.to_numpy()[np.arange(len(df)), true_idx],
                                 index=df.index)
    return pd.DataFrame(labels), pd.DataFrame(p_true), truth, classes


def stability_table(labels: pd.DataFrame, p_true: pd.DataFrame,
                    truth: pd.Series, classes: np.ndarray,
                    target: str) -> pd.DataFrame:
    """Per-sample label counts, agreement, share correct and mean true prob."""
    out = pd.DataFrame(index=labels.index)
    for c in classes:
        out[f"{target}_n_{c}"] = (labels == c).sum(axis=1)
    counts = out[[f"{target}_n_{c}" for c in classes]]
    out[f"{target}_agreement"] = counts.max(axis=1) / labels.shape[1]
    out[f"{target}_correct"] = labels.eq(truth.loc[labels.index],
                                         axis=0).mean(axis=1)
    out[f"{target}_prob_true"] = p_true.mean(axis=1)
    return out


def run_tests(stab: pd.DataFrame) -> pd.DataFrame:
    """Compare Tumor / Normal stability with each cluster target's stability."""
    rows = []
    for target in CLUSTER_TARGETS:
        for measure in ["agreement", "prob_true"]:
            a = stab[f"group_{measure}"]
            b = stab[f"{target}_{measure}"]
            if a.nunique() < 2 or b.nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(a, b)
            rows.append({"cluster_target": target, "test": "spearman",
                         "measure": measure, "statistic": rho, "pvalue": p,
                         "n_samples": len(a)})

        # 2x2 table: fully stable (all models agree) vs not, per target.
        stable_a = stab["group_agreement"] == 1
        stable_b = stab[f"{target}_agreement"] == 1
        table = pd.crosstab(stable_a, stable_b).reindex(
            index=[True, False], columns=[True, False], fill_value=0)
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            odds, p = np.nan, np.nan
        else:
            odds, p = fisher_exact(table.to_numpy())
        rows.append({"cluster_target": target, "test": "fisher_exact",
                     "measure": "agreement == 1", "statistic": odds,
                     "pvalue": p, "n_samples": int(table.to_numpy().sum()),
                     "table_stable_stable": table.loc[True, True],
                     "table_stable_unstable": table.loc[True, False],
                     "table_unstable_stable": table.loc[False, True],
                     "table_unstable_unstable": table.loc[False, False]})

    tests = pd.DataFrame(rows)
    tested = tests["pvalue"].notna()
    tests["padj"] = np.nan
    if tested.any():
        tests.loc[tested, "padj"] = multipletests(
            tests.loc[tested, "pvalue"], method="fdr_bh")[1]
    return tests


def plot_matrix(correct: pd.DataFrame, groups: pd.Series, labels: pd.Series,
                out_path: Path) -> None:
    """Samples x models grid: is the model's label the sample's true label?"""
    # Order samples by group, then by how often they are predicted correctly.
    order = (pd.DataFrame({"group": groups, "score": correct.mean(axis=1)})
             .sort_values(["group", "score"], ascending=[True, False]).index)
    mat = correct.loc[order]

    fig, (ax_bar, ax) = plt.subplots(
        1, 2, figsize=(11, 12), gridspec_kw={"width_ratios": [0.03, 1]},
        sharey=True)
    bar = np.array([[0 if g == "Normal" else 1] for g in groups.loc[order]])
    ax_bar.imshow(bar, aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([GROUP_COLORS["Normal"],
                                       GROUP_COLORS["Tumor"]]))
    ax_bar.set_xticks([])
    ax_bar.set_ylabel(f"Samples (n={len(order)}), sorted by group")

    ax.imshow(mat.to_numpy().astype(int), aspect="auto",
              interpolation="nearest", cmap=ListedColormap(CORRECT_COLORS),
              vmin=0, vmax=1)
    ax.set_xticks(range(mat.shape[1]),
                  [c.split(":")[1] for c in mat.columns], rotation=90,
                  fontsize=7)
    # Separate the three targets and name them above the grid.
    n_per = mat.shape[1] // len(TARGETS)
    for i, target in enumerate(TARGETS):
        if i:
            ax.axvline(i * n_per - 0.5, color="black", lw=1.2)
        ax.text((i + 0.5) * n_per - 0.5, -1.5, TARGET_NAMES[target],
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Model (method_number of genes)")
    ax.tick_params(axis="y", labelleft=False)

    # Only name the samples that at least one model gets wrong.
    wrong = [i for i, s in enumerate(order) if not mat.loc[s].all()]
    ax_bar.set_yticks(wrong, [labels[order[i]] for i in wrong], fontsize=7)

    handles = [mpatches.Patch(color=CORRECT_COLORS[1], label="Correct"),
               mpatches.Patch(color=CORRECT_COLORS[0], label="Wrong"),
               mpatches.Patch(color=GROUP_COLORS["Normal"], label="Normal"),
               mpatches.Patch(color=GROUP_COLORS["Tumor"], label="Tumor")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              title="Prediction / group")
    fig.suptitle("Sample x model predictions (label = repeat-averaged "
                 "out-of-fold probability)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_stability(stab: pd.DataFrame, tests: pd.DataFrame,
                   groups: pd.Series, labels: pd.Series,
                   out_path: Path) -> None:
    """Mean true-label probability: Tumor / Normal vs cluster, per cluster target."""
    fig, axes = plt.subplots(1, len(CLUSTER_TARGETS), figsize=(12, 5.5))
    for ax, target in zip(axes, CLUSTER_TARGETS):
        x = stab["group_prob_true"]
        y = stab[f"{target}_prob_true"]
        for group, color in GROUP_COLORS.items():
            m = groups == group
            ax.scatter(x[m], y[m], c=color, s=40, alpha=0.8,
                       edgecolors="white", linewidths=0.5, label=group)
        for s in stab.index[(x < LOW_PROB) | (y < LOW_PROB)]:
            ax.annotate(labels[s], (x[s], y[s]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        t = tests[(tests["cluster_target"] == target)
                  & (tests["test"] == "spearman")
                  & (tests["measure"] == "prob_true")].iloc[0]
        ax.set_title(f"{TARGET_NAMES[target]}\nSpearman rho = "
                     f"{t['statistic']:.2f}, BH padj = {t['padj']:.2g}")
        ax.set_xlabel("Mean probability of true group (Tumor vs Normal models)")
        ax.set_ylabel(f"Mean probability of own cluster ({TARGET_NAMES[target]})")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].legend(title="Group", loc="lower right")
    fig.suptitle("Per-sample prediction stability across the 10 models of "
                 "each target")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    preds, labels = load_inputs()

    matrices, correct, stabs, truths = [], [], [], {}
    for target in TARGETS:
        lab, p_true, truth, classes = model_labels(preds, target)
        truths[target] = truth
        matrices.append(lab.add_prefix(f"{target}:"))
        correct.append(lab.eq(truth.loc[lab.index], axis=0)
                       .add_prefix(f"{target}:"))
        stabs.append(stability_table(lab, p_true, truth, classes, target))

    truth_cols = pd.DataFrame({f"true_{t}": truths[t] for t in TARGETS})
    samples = truth_cols.index
    groups = truths["group"]

    matrix = pd.concat([truth_cols] + matrices, axis=1)
    matrix.insert(0, "label", labels.loc[samples])
    matrix.index.name = "sample"
    out = TBL_DIR / "sample_by_model.csv"
    matrix.to_csv(out)
    print(f"Saved samples x models matrix {matrix.shape} -> {out}")

    stab = pd.concat([truth_cols] + stabs, axis=1)
    stab.insert(0, "label", labels.loc[samples])
    stab.index.name = "sample"
    out = TBL_DIR / "sample_stability.csv"
    stab.to_csv(out)
    print(f"Saved per-sample stability -> {out}")

    unstable = stab[(stab[[f"{t}_agreement" for t in TARGETS]] < 1).any(axis=1)
                    | (stab[[f"{t}_correct" for t in TARGETS]] < 1).any(axis=1)]
    cols = ["label"] + [f"{t}_{m}" for t in TARGETS
                        for m in ["correct", "prob_true"]]
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.2f}".format):
        print(f"\nSamples not predicted correctly by every model "
              f"({len(unstable)}):")
        print(unstable[cols].to_string())

    tests = run_tests(stab)
    out = TBL_DIR / "stability_tests.csv"
    tests.to_csv(out, index=False)
    print(f"\nSaved {len(tests)} tests -> {out}")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.3g}".format):
        print(tests.to_string(index=False))

    plot_matrix(pd.concat(correct, axis=1), groups, labels,
                FIG_DIR / "12_sample_by_model.png")
    plot_stability(stab, tests, groups, labels, FIG_DIR / "12_stability.png")


if __name__ == "__main__":
    main()
