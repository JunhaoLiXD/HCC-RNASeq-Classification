"""
Gene signatures of the Tumor vs Normal models and a heatmap of those genes.

Both models from Step 11 are refitted on all 100 samples with the top 5,000
most variable genes (same filtering, log transform and z-scoring). The
signature of each model is its top 50 genes: largest |coefficient| for
logistic regression and largest impurity importance for random forest. The
union of the two signatures is drawn as a clustermap (average linkage,
1 - Pearson r on rows and columns) with Tumor / Normal and K-means (k = 3)
annotation bars, and a row bar showing which signature each gene comes from.

Signature genes are also compared with the Assignment 2 DESeq2 results.

Writes signature_genes.csv and 13_signature_heatmap.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "assn4" / "figures"
TBL_DIR = ROOT / "results" / "assn4" / "tables"
ASSN2_TBL_DIR = ROOT / "results" / "assn2" / "tables"   # sample sheet, DEGs
ASSN3_TBL_DIR = ROOT / "results" / "assn3" / "tables"   # kmeans_assignments.csv

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}
KMEANS_COLORS = {1: "#2ca02c", 2: "#ff7f0e", 3: "#8c564b"}
SIGNATURE_COLORS = {"LR only": "#4c72b0", "RF only": "#dd8452",
                    "Both": "#55a868"}

MIN_COUNT = 10          # keep a gene if its count exceeds this
MIN_SAMPLES = 10        # in at least this many samples
N_MAIN = 5000           # genes the models are trained on
N_SIGNATURE = 50        # top genes taken from each model
KMEANS_COL = "kmeans_g5000_k3"
ALPHA = 0.05            # DESeq2 padj threshold for the DEG comparison
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "correlation"
Z_LIMIT = 3             # color scale limits for the z-scores
# Same model settings as Step 11.
RF_TREES = 500
LR_MAX_ITER = 5000
RANDOM_STATE = 42


def load_inputs():
    """Load log2 expression of the top-variance genes and the annotations."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(ASSN2_TBL_DIR / "sample_sheet.csv", index_col=0)
    kmeans = pd.read_csv(ASSN3_TBL_DIR / "kmeans_assignments.csv", index_col=0)
    annotation = pd.read_csv(ASSN2_TBL_DIR / "gene_annotation.csv", index_col=0)
    diffexp = pd.read_csv(ASSN2_TBL_DIR / "diffexp_full.csv", index_col=0)
    strict = pd.read_csv(ASSN2_TBL_DIR / "significant_degs.csv", index_col=0)

    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    log_expr = np.log2(expr.loc[keep] + 1.0)
    top_genes = log_expr.var(axis=1).nlargest(N_MAIN).index
    log_expr = log_expr.loc[top_genes]

    samples = log_expr.columns
    groups = sample_sheet.loc[samples, "group"]
    clusters = kmeans.loc[samples, KMEANS_COL]
    return log_expr, groups, clusters, annotation, diffexp, strict


def fit_signatures(log_expr: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Refit both models on all samples; return per-gene weights and ranks."""
    x = StandardScaler().fit_transform(log_expr.T)
    y = (groups == "Tumor").to_numpy()

    lr = LogisticRegression(C=1.0, max_iter=LR_MAX_ITER).fit(x, y)
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_features="sqrt", n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(x, y)

    genes = pd.DataFrame(index=log_expr.index)
    genes["lr_coef"] = lr.coef_[0]          # > 0: higher in Tumor
    genes["rf_importance"] = rf.feature_importances_
    genes["lr_rank"] = genes["lr_coef"].abs().rank(ascending=False,
                                                    method="first").astype(int)
    genes["rf_rank"] = genes["rf_importance"].rank(ascending=False,
                                                   method="first").astype(int)
    in_lr = genes["lr_rank"] <= N_SIGNATURE
    in_rf = genes["rf_rank"] <= N_SIGNATURE
    genes["signature"] = np.select(
        [in_lr & in_rf, in_lr, in_rf], ["Both", "LR only", "RF only"],
        default="")
    return genes


def main() -> None:
    log_expr, groups, clusters, annotation, diffexp, strict = load_inputs()
    genes = fit_signatures(log_expr, groups)

    sig = genes[genes["signature"] != ""].copy()
    sig = sig.join(annotation["gene_symbol"])
    sig = sig.join(diffexp[["log2FoldChange", "padj"]])
    sig["deg_padj05"] = sig["padj"] < ALPHA
    sig["deg_strict"] = sig.index.isin(strict.index)
    sig = sig.sort_values(["signature", "lr_rank"])
    sig.index.name = "ensembl_id"
    out = TBL_DIR / "signature_genes.csv"
    sig.to_csv(out)
    print(f"Signature genes: {len(sig)} "
          f"({sig['signature'].value_counts().to_dict()}) -> {out}")

    # Overlap with the Assignment 2 DEGs, against the 5,000-gene background.
    bg = diffexp.reindex(log_expr.index)
    bg_deg = (bg["padj"] < ALPHA).mean()
    bg_strict = log_expr.index.isin(strict.index).mean()
    print(f"DEG (padj<{ALPHA}) share: signature {sig['deg_padj05'].mean():.1%}"
          f" vs top-{N_MAIN} background {bg_deg:.1%}")
    print(f"Strict DEG share        : signature {sig['deg_strict'].mean():.1%}"
          f" vs top-{N_MAIN} background {bg_strict:.1%}")
    agree = np.sign(sig["lr_coef"]) == np.sign(sig["log2FoldChange"])
    print(f"LR coefficient sign matches DESeq2 log2FC sign: "
          f"{int(agree.sum())} / {len(sig)}")

    # --- heatmap ------------------------------------------------------------
    mat = log_expr.loc[sig.index]
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    labels = sig["gene_symbol"].fillna(sig.index.to_series())
    z.index = labels.to_numpy()

    row_linkage = linkage(z.to_numpy(), method=LINKAGE_METHOD,
                          metric=DISTANCE_METRIC)
    col_linkage = linkage(z.T.to_numpy(), method=LINKAGE_METHOD,
                          metric=DISTANCE_METRIC)
    col_colors = pd.DataFrame({
        "Group": groups.map(GROUP_COLORS),
        "K-means (k=3)": clusters.map(KMEANS_COLORS),
    }, index=z.columns)
    row_colors = pd.Series(sig["signature"].map(SIGNATURE_COLORS).to_numpy(),
                           index=z.index, name="Signature")

    g = sns.clustermap(
        z,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        col_colors=col_colors,
        row_colors=row_colors,
        cmap="RdBu_r",
        center=0, vmin=-Z_LIMIT, vmax=Z_LIMIT,
        xticklabels=False, yticklabels=True,
        figsize=(13, 16),
        dendrogram_ratio=(0.12, 0.10),
        colors_ratio=0.015,
        cbar_pos=(0.04, 0.865, 0.02, 0.075),
        cbar_kws={"label": "Gene z-score"},
    )
    g.ax_heatmap.tick_params(axis="y", labelsize=6)
    g.ax_heatmap.set_xlabel(f"Samples (n={z.shape[1]})")
    g.ax_heatmap.set_ylabel(f"Signature genes (n={z.shape[0]}; top "
                            f"{N_SIGNATURE} per model)")

    # Figure-level legends, stacked to the right of the gene labels.
    present = set(sig["signature"])
    legends = [
        ("Group", GROUP_COLORS),
        ("K-means (k=3)", {f"Cluster {c}": col
                           for c, col in KMEANS_COLORS.items()}),
        ("Signature", {l: c for l, c in SIGNATURE_COLORS.items()
                       if l in present}),
    ]
    y = 0.84
    for title, mapping in legends:
        handles = [mpatches.Patch(color=c, label=l) for l, c in mapping.items()]
        g.figure.legend(
            handles=handles, title=title, loc="upper left",
            bbox_to_anchor=(1.0, y), frameon=True, fontsize=9,
            title_fontsize=9,
        )
        y -= 0.025 * (len(mapping) + 2)

    g.figure.suptitle(
        f"Tumor vs Normal model signatures: top {N_SIGNATURE} logistic "
        f"regression and random forest genes\n"
        f"(rows and columns: {LINKAGE_METHOD} linkage, 1 - Pearson r)",
        y=1.03, fontsize=13,
    )
    out = FIG_DIR / "13_signature_heatmap.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved heatmap -> {out}")


if __name__ == "__main__":
    main()
