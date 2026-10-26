"""
Supervised classification of the samples with logistic regression and random
forest, evaluated by repeated patient-grouped cross-validation.

Three targets are predicted:
  1. group      - Tumor vs Normal (Assignment 1 groups)
  2. kmeans_k2  - K-means clusters at k = 2 (Assignment 3, top 5,000 genes)
  3. kmeans_k3  - K-means clusters at k = 3 (multiclass; both models handle
                  more than two classes natively, so no one-vs-rest split)

Features follow Step 2: low-count filter and log2(count + 1) on all samples
(label-free). Inside every training fold the top-N most variable genes are
selected and z-scored, so the held-out fold never informs the features.
N = 10 / 100 / 1,000 / 5,000 / 10,000 genes.

Each tumor and its matched normal come from the same patient, so folds are
split by patient (StratifiedGroupKFold) and both samples of a patient are
always held out together. The 5-fold CV is repeated with different seeds;
every repeat gives one out-of-fold prediction per sample.

Writes oof_predictions.csv, model_performance.csv, 11_auc_vs_genes.png and
11_roc_curves.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
META_PATH = DATA_DIR / "metadata_SRP068976.tsv"
FIG_DIR = ROOT / "results" / "assn4" / "figures"
TBL_DIR = ROOT / "results" / "assn4" / "tables"
ASSN2_TBL_DIR = ROOT / "results" / "assn2" / "tables"   # sample_sheet.csv (Step 1)
ASSN3_TBL_DIR = ROOT / "results" / "assn3" / "tables"   # kmeans_assignments.csv
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {"logreg": "#4c72b0", "rf": "#dd8452"}
METHOD_NAMES = {"logreg": "Logistic regression", "rf": "Random forest"}

MIN_COUNT = 10          # keep a gene if its count exceeds this
MIN_SAMPLES = 10        # in at least this many samples
GENE_COUNTS = [10, 100, 1000, 5000, 10000]
N_MAIN = 5000           # gene number used for the ROC curves
TARGETS = {             # target name -> column in kmeans_assignments.csv
    "group": None,
    "kmeans_k2": "kmeans_g5000_k2",
    "kmeans_k3": "kmeans_g5000_k3",
}
TARGET_NAMES = {
    "group": "Tumor vs Normal",
    "kmeans_k2": "K-means clusters (k = 2)",
    "kmeans_k3": "K-means clusters (k = 3)",
}
N_FOLDS = 5
N_REPEATS = 10
RF_TREES = 500
LR_MAX_ITER = 5000
RANDOM_STATE = 42


class TopVarianceGenes(BaseEstimator, TransformerMixin):
    """Keep the n_genes columns with the highest variance in the training data."""

    def __init__(self, n_genes: int = 5000):
        self.n_genes = n_genes

    def fit(self, x, y=None):
        var = np.asarray(x).var(axis=0, ddof=1)
        self.idx_ = np.sort(np.argsort(var)[::-1][:self.n_genes])
        return self

    def transform(self, x):
        return np.asarray(x)[:, self.idx_]


def make_model(method: str, n_genes: int) -> Pipeline:
    """Gene selection -> z-scoring -> classifier, fitted together per fold."""
    if method == "logreg":
        # L2 penalty (the default, l1_ratio = 0) with C = 1.0; needed because
        # genes far outnumber samples and the classes are nearly separable.
        clf = LogisticRegression(C=1.0, max_iter=LR_MAX_ITER)
    else:
        clf = RandomForestClassifier(
            n_estimators=RF_TREES, max_features="sqrt", n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    return Pipeline([
        ("select", TopVarianceGenes(n_genes)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def load_inputs():
    """Return log2 features (samples x genes), the targets and patient ids."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(ASSN2_TBL_DIR / "sample_sheet.csv", index_col=0)
    kmeans = pd.read_csv(ASSN3_TBL_DIR / "kmeans_assignments.csv", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t", index_col=0)
    samples = expr.columns

    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    log_expr = np.log2(expr.loc[keep] + 1.0)
    print(f"Genes kept after low-count filter : {int(keep.sum())}")

    targets = pd.DataFrame(index=samples)
    for name, col in TARGETS.items():
        if col is None:
            targets[name] = sample_sheet.loc[samples, "group"]
        else:
            targets[name] = kmeans.loc[samples, col].astype(str)

    # Titles look like "Patient 39 Tumor"; the patient number is the group id.
    patient = meta.loc[samples, "refinebio_title"].str.extract(
        r"Patient (\d+)")[0].astype(int)
    assert patient.value_counts().eq(2).all(), "Expected 2 samples per patient"
    return log_expr.T, targets, patient


def brier_score(y_true: np.ndarray, proba: np.ndarray,
                classes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot label."""
    onehot = (y_true[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def auc_score(y_true: np.ndarray, proba: np.ndarray,
              classes: np.ndarray) -> float:
    """Binary AUC, or macro one-vs-rest AUC for more than two classes."""
    if len(classes) == 2:
        return roc_auc_score(y_true == classes[1], proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                         labels=classes)


def run_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str,
           n_genes: int, classes: np.ndarray):
    """Repeated grouped CV; returns out-of-fold probabilities per repeat."""
    oof = np.zeros((N_REPEATS, len(y), len(classes)))
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=RANDOM_STATE + rep)
        for train, test in cv.split(x, y, groups):
            model = make_model(method, n_genes).fit(x[train], y[train])
            # Map the fold's class order onto the full class list.
            cols = np.searchsorted(classes, model.classes_)
            oof[rep][np.ix_(test, cols)] = model.predict_proba(x[test])
    return oof


def plot_auc_vs_genes(perf: pd.DataFrame, out_path: Path) -> None:
    """Mean CV AUC (+/- SD over repeats) vs number of genes, one panel per target."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 4.5),
                             sharey=True)
    for ax, target in zip(axes, TARGETS):
        sub = perf[perf["target"] == target]
        for method, color in METHOD_COLORS.items():
            m = sub[sub["method"] == method].sort_values("n_genes")
            ax.errorbar(m["n_genes"], m["auc_mean"], yerr=m["auc_sd"],
                        marker="o", capsize=3, color=color,
                        label=METHOD_NAMES[method])
        ax.set_xscale("log")
        ax.set_xticks(GENE_COUNTS, [f"{n // 1000}k" if n >= 1000 else str(n)
                                    for n in GENE_COUNTS])
        ax.minorticks_off()
        ax.set_xlabel("Number of most variable genes")
        ax.set_title(TARGET_NAMES[target])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-validated AUC (mean ± SD)")
    axes[-1].legend(title="Method", loc="lower right")
    fig.suptitle(f"Model performance vs number of genes "
                 f"({N_FOLDS}-fold patient-grouped CV x {N_REPEATS} repeats; "
                 f"k = 3 uses macro one-vs-rest AUC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_roc_curves(oof_store: dict, y: np.ndarray, out_path: Path) -> None:
    """Tumor vs Normal ROC curves at N_MAIN genes, one thin line per repeat."""
    fig, ax = plt.subplots(figsize=(6, 6))
    positive = y == "Tumor"
    for method, color in METHOD_COLORS.items():
        oof = oof_store[(method, N_MAIN)]   # repeats x samples x [Normal, Tumor]
        mean_auc = np.mean([roc_auc_score(positive, oof[rep][:, 1])
                            for rep in range(N_REPEATS)])
        for rep in range(N_REPEATS):
            fpr, tpr, _ = roc_curve(positive, oof[rep][:, 1])
            label = (f"{METHOD_NAMES[method]} (mean AUC {mean_auc:.3f})"
                     if rep == 0 else None)
            ax.plot(fpr, tpr, color=color, alpha=0.6, lw=1.2, label=label)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate (Normal called Tumor)")
    ax.set_ylabel("True positive rate (Tumor called Tumor)")
    ax.set_title(f"Tumor vs Normal ROC, top {N_MAIN:,} genes\n"
                 f"(out-of-fold, one line per CV repeat)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    features, targets, patient = load_inputs()
    x = features.to_numpy()
    groups = patient.to_numpy()
    samples = features.index

    pred_rows = []
    perf_rows = []
    group_oof = {}
    for target in TARGETS:
        y = targets[target].to_numpy()
        classes = np.unique(y)
        print(f"\n=== Target: {target} "
              f"({dict(zip(*np.unique(y, return_counts=True)))}) ===")
        for method in METHOD_COLORS:
            for n_genes in GENE_COUNTS:
                oof = run_cv(x, y, groups, method, n_genes, classes)
                if target == "group":
                    group_oof[(method, n_genes)] = oof

                aucs, accs, briers = [], [], []
                for rep in range(N_REPEATS):
                    pred = classes[oof[rep].argmax(axis=1)]
                    aucs.append(auc_score(y, oof[rep], classes))
                    accs.append(accuracy_score(y, pred))
                    briers.append(brier_score(y, oof[rep], classes))
                    rows = pd.DataFrame(
                        oof[rep], columns=[f"prob_{c}" for c in classes])
                    rows.insert(0, "pred", pred)
                    rows.insert(0, "true", y)
                    rows.insert(0, "repeat", rep)
                    rows.insert(0, "n_genes", n_genes)
                    rows.insert(0, "method", method)
                    rows.insert(0, "target", target)
                    rows.insert(0, "sample", samples)
                    pred_rows.append(rows)

                perf_rows.append({
                    "target": target, "method": method, "n_genes": n_genes,
                    "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs, ddof=1),
                    "accuracy_mean": np.mean(accs),
                    "accuracy_sd": np.std(accs, ddof=1),
                    "brier_mean": np.mean(briers),
                    "brier_sd": np.std(briers, ddof=1),
                })
                p = perf_rows[-1]
                print(f"{method:6s} {n_genes:>6,} genes  "
                      f"AUC={p['auc_mean']:.4f}±{p['auc_sd']:.4f}  "
                      f"acc={p['accuracy_mean']:.3f}  "
                      f"brier={p['brier_mean']:.4f}")

    preds = pd.concat(pred_rows, ignore_index=True)
    out = TBL_DIR / "oof_predictions.csv"
    preds.to_csv(out, index=False)
    print(f"\nSaved {len(preds):,} out-of-fold predictions -> {out}")

    perf = pd.DataFrame(perf_rows)
    out = TBL_DIR / "model_performance.csv"
    perf.to_csv(out, index=False)
    print(f"Saved model performance -> {out}")

    plot_auc_vs_genes(perf, FIG_DIR / "11_auc_vs_genes.png")
    plot_roc_curves(group_oof, targets["group"].to_numpy(),
                    FIG_DIR / "11_roc_curves.png")


if __name__ == "__main__":
    main()
