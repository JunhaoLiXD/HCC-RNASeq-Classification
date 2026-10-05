"""
Heatmap of the 5,000 most variable genes used for clustering, with row and
column dendrograms and three annotation bars: K-means clusters, hierarchical
clusters (both at the primary k) and the Tumor / Normal groups.

The column dendrogram is the same average-linkage / correlation-distance tree
used in Step 8, recomputed here and checked against hclust_assignments.csv.
Genes are clustered the same way. Values are per-gene z-scores, clipped to
+/-3 for the color scale only.

Writes 09_cluster_heatmap.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "assn3" / "figures"
TBL_DIR = ROOT / "results" / "assn3" / "tables"
ASSN2_TBL_DIR = ROOT / "results" / "assn2" / "tables"   # sample_sheet.csv (Step 1)

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}
# Cluster colors are kept distinct from the group colors.
KMEANS_COLORS = {1: "#2ca02c", 2: "#ff7f0e"}
HCLUST_COLORS = {1: "#9467bd", 2: "#bcbd22"}

MIN_COUNT = 10          # keep a gene if its count exceeds this
MIN_SAMPLES = 10        # in at least this many samples
N_MAIN = 5000           # genes shown in the heatmap
PRIMARY_K = 2
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "correlation"
Z_LIMIT = 3             # color scale limits for the z-scores


def build_feature_matrix(expr: pd.DataFrame) -> pd.DataFrame:
    """Return a samples x genes matrix of the top-variance genes, z-scored."""
    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    log_expr = np.log2(expr.loc[keep] + 1.0)
    top_genes = log_expr.var(axis=1).nlargest(N_MAIN).index
    x = log_expr.loc[top_genes].T
    x_scaled = StandardScaler().fit_transform(x)
    return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)


def main() -> None:
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(ASSN2_TBL_DIR / "sample_sheet.csv", index_col=0)
    sample_sheet = sample_sheet.loc[expr.columns]
    kmeans = pd.read_csv(TBL_DIR / "kmeans_assignments.csv", index_col=0)
    hclust = pd.read_csv(TBL_DIR / "hclust_assignments.csv", index_col=0)

    features = build_feature_matrix(expr)
    samples = features.index
    km_labels = kmeans.loc[samples, f"kmeans_g{N_MAIN}_k{PRIMARY_K}"]
    hc_labels = hclust.loc[samples, f"hclust_g{N_MAIN}_k{PRIMARY_K}"]
    groups = sample_sheet.loc[samples, "group"]

    col_linkage = linkage(features.to_numpy(), method=LINKAGE_METHOD,
                          metric=DISTANCE_METRIC)
    # The recomputed tree must reproduce the Step 8 clusters exactly.
    recut = fcluster(col_linkage, t=PRIMARY_K, criterion="maxclust")
    assert adjusted_rand_score(recut, hc_labels) == 1.0, \
        "Column linkage does not match hclust_assignments.csv"

    row_linkage = linkage(features.T.to_numpy(), method=LINKAGE_METHOD,
                          metric=DISTANCE_METRIC)
    print(f"Heatmap matrix: {features.shape[1]} genes x {features.shape[0]} "
          f"samples")

    col_colors = pd.DataFrame({
        f"K-means (k={PRIMARY_K})": km_labels.map(KMEANS_COLORS),
        f"Hierarchical (k={PRIMARY_K})": hc_labels.map(HCLUST_COLORS),
        "Group": groups.map(GROUP_COLORS),
    }, index=samples)

    g = sns.clustermap(
        features.T,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        col_colors=col_colors,
        cmap="RdBu_r",
        center=0, vmin=-Z_LIMIT, vmax=Z_LIMIT,
        xticklabels=False, yticklabels=False,
        figsize=(12, 13),
        dendrogram_ratio=(0.12, 0.14),
        colors_ratio=0.02,
        cbar_pos=(0.02, 0.80, 0.025, 0.12),
        cbar_kws={"label": "Gene z-score"},
        rasterized=True,
    )
    g.ax_heatmap.set_xlabel(f"Samples (n={features.shape[0]})")
    g.ax_heatmap.set_ylabel(f"Top {N_MAIN:,} most variable genes")

    # One legend block per annotation bar, stacked to the right of the heatmap.
    legends = [
        (f"K-means (k={PRIMARY_K})",
         {f"Cluster {c}": col for c, col in KMEANS_COLORS.items()}),
        (f"Hierarchical (k={PRIMARY_K})",
         {f"Cluster {c}": col for c, col in HCLUST_COLORS.items()}),
        ("Group", GROUP_COLORS),
    ]
    y = 1.0
    for title, mapping in legends:
        handles = [mpatches.Patch(color=c, label=l) for l, c in mapping.items()]
        leg = g.ax_heatmap.legend(
            handles=handles, title=title, loc="upper left",
            bbox_to_anchor=(1.01, y), frameon=True, fontsize=9,
            title_fontsize=9,
        )
        g.ax_heatmap.add_artist(leg)
        y -= 0.13

    g.figure.suptitle(
        f"Top {N_MAIN:,} variable genes: K-means and hierarchical clusters "
        f"(k={PRIMARY_K}) vs Tumor / Normal\n"
        f"(rows and columns: {LINKAGE_METHOD} linkage, 1 - Pearson r)",
        y=1.04, fontsize=13,
    )

    out = FIG_DIR / "09_cluster_heatmap.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved heatmap -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
