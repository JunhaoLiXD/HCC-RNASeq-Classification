"""
Hierarchical clustering of the samples: average linkage on correlation
distance (1 - Pearson r), cut into k clusters with fcluster(maxclust).

Features follow Step 2: low-count filter, log2(count + 1), top-N most variable
genes, z-scored per gene. Two experiments are run:

  1. k sweep: k = 2..6 on the top 5,000 genes, with the silhouette width on
     the same correlation distance to guide the choice of k.
  2. Gene-number sweep: the primary k on the top 10 / 100 / 1,000 / 5,000 /
     10,000 genes.

Average linkage can split off single outlying samples, so every cut reports
its cluster sizes and lists the members of small clusters. Cluster labels are
renumbered 1..k by decreasing cluster size.

Writes hclust_assignments.csv, hclust_k_summary.csv, 08_hclust_k_selection.png,
08_hclust_membership_by_k.png and 08_hclust_dendrogram.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "assn3" / "figures"
TBL_DIR = ROOT / "results" / "assn3" / "tables"
ASSN2_TBL_DIR = ROOT / "results" / "assn2" / "tables"   # sample_sheet.csv (Step 1)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}

MIN_COUNT = 10          # keep a gene if its count exceeds this
MIN_SAMPLES = 10        # in at least this many samples
N_MAIN = 5000           # genes used for the main analysis and the k sweep
K_RANGE = range(2, 7)   # k values compared on the main gene set
PRIMARY_K = 2           # k used for the gene-number sweep and the heatmap
GENE_COUNTS = [10, 100, 1000, 5000, 10000]
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "correlation"   # 1 - Pearson r between samples
SMALL_CLUSTER = 3       # list the members of clusters this size or smaller
METHOD = "hclust"


def load_inputs():
    """Load the expression matrix and the sample sheet built in Step 1."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(ASSN2_TBL_DIR / "sample_sheet.csv", index_col=0)
    sample_sheet = sample_sheet.loc[expr.columns]
    return expr, sample_sheet


def filter_and_log(expr: pd.DataFrame) -> pd.DataFrame:
    """Drop lowly expressed genes and log2-transform the counts."""
    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    print(f"Genes kept after low-count filter : {int(keep.sum())}")
    return np.log2(expr.loc[keep] + 1.0)


def build_feature_matrix(log_expr: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    """Return a samples x genes matrix of the top-variance genes, z-scored."""
    top_genes = log_expr.var(axis=1).nlargest(n_genes).index
    x = log_expr.loc[top_genes].T
    x_scaled = StandardScaler().fit_transform(x)
    return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster ids 1..k by decreasing size (ties: first appearance)."""
    ids, first_idx, counts = np.unique(labels, return_index=True,
                                       return_counts=True)
    order = sorted(range(len(ids)), key=lambda i: (-counts[i], first_idx[i]))
    mapping = {ids[i]: rank + 1 for rank, i in enumerate(order)}
    return np.array([mapping[l] for l in labels])


def cut_tree(z: np.ndarray, k: int) -> np.ndarray:
    """Cut the linkage into k clusters and return size-ordered labels."""
    labels = fcluster(z, t=k, criterion="maxclust")
    n_found = len(np.unique(labels))
    if n_found != k:
        print(f"  WARNING: requested k={k} but the cut gave {n_found} clusters")
    return relabel_by_size(labels)


def report_small_clusters(labels: pd.Series, groups: pd.Series) -> None:
    """Print the members of very small clusters so outliers are visible."""
    sizes = labels.value_counts()
    for cl in sizes[sizes <= SMALL_CLUSTER].index:
        members = labels.index[labels == cl]
        desc = ", ".join(f"{s} ({groups[s]})" for s in members)
        print(f"  small cluster {cl} (n={len(members)}): {desc}")


def summarize_clusters(labels: pd.Series, groups: pd.Series, k: int,
                       n_genes: int) -> pd.DataFrame:
    """One row per cluster with its size and Tumor / Normal counts."""
    tab = pd.crosstab(labels, groups).reindex(columns=["Tumor", "Normal"],
                                              fill_value=0)
    tab = tab.rename(columns={"Tumor": "n_tumor", "Normal": "n_normal"})
    tab.insert(0, "n_samples", tab.sum(axis=1))
    tab.index.name = "cluster"
    tab = tab.reset_index()
    tab.insert(0, "k", k)
    tab.insert(0, "n_genes", n_genes)
    tab.insert(0, "method", METHOD)
    return tab


def plot_k_selection(scores: pd.DataFrame, out_path: Path) -> None:
    """Silhouette (correlation distance) over k."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(scores["k"], scores["silhouette"], marker="o", color="#4c72b0")
    for _, row in scores.iterrows():
        ax.annotate(f"min size {int(row['min_cluster_size'])}",
                    (row["k"], row["silhouette"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.axvline(PRIMARY_K, color="grey", ls="--", lw=0.8,
               label=f"Primary k = {PRIMARY_K}")
    ax.set_xticks(list(K_RANGE))
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Mean silhouette width (1 - Pearson r)")
    ax.set_title(f"Hierarchical clustering k selection\n"
                 f"({LINKAGE_METHOD} linkage, top {N_MAIN:,} variable genes)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_membership(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    """Stacked Tumor / Normal bars per cluster, one panel per k."""
    ks = sorted(summary["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 4.5),
                             sharey=True)
    for ax, k in zip(axes, ks):
        sub = summary[summary["k"] == k]
        x = np.arange(len(sub))
        ax.bar(x, sub["n_tumor"], color=GROUP_COLORS["Tumor"])
        ax.bar(x, sub["n_normal"], bottom=sub["n_tumor"],
               color=GROUP_COLORS["Normal"])
        for xi, n in zip(x, sub["n_samples"]):
            ax.text(xi, n + 1, str(n), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, sub["cluster"].astype(str))
        ax.set_xlabel("Cluster")
        ax.set_title(f"k = {k}")
    axes[0].set_ylabel("Number of samples")
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    axes[-1].legend(handles=handles, title="Group", loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_dendrogram(z: np.ndarray, samples: pd.Index, groups: pd.Series,
                    out_path: Path) -> None:
    """Sample dendrogram with the primary-k cut and group-colored leaves."""
    # Any height between the merges that leave k and k-1 clusters gives the
    # same maxclust cut; draw the line halfway between them.
    cut_height = (z[-PRIMARY_K, 2] + z[-(PRIMARY_K - 1), 2]) / 2

    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(
        z, labels=list(samples), ax=ax, color_threshold=cut_height,
        above_threshold_color="grey", leaf_font_size=6,
    )
    for tick in ax.get_xticklabels():
        tick.set_color(GROUP_COLORS[groups[tick.get_text()]])
    ax.axhline(cut_height, color="black", ls="--", lw=0.8)

    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    handles.append(plt.Line2D([], [], color="black", ls="--",
                              label=f"Cut for k = {PRIMARY_K}"))
    ax.legend(handles=handles, title="Leaf label color", loc="upper right")
    ax.set_xlabel("Samples (labels colored by group)")
    ax.set_ylabel("Linkage distance (1 - Pearson r)")
    ax.set_title(f"Hierarchical clustering of samples ({LINKAGE_METHOD} "
                 f"linkage, correlation distance, top {N_MAIN:,} genes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    expr, sample_sheet = load_inputs()
    groups = sample_sheet["group"]
    log_expr = filter_and_log(expr)

    assignments = pd.DataFrame(index=log_expr.columns)
    assignments.index.name = "sample"
    summaries = []

    # --- k sweep on the main gene set ----------------------------------------
    x_main = build_feature_matrix(log_expr, N_MAIN).to_numpy()
    z_main = linkage(x_main, method=LINKAGE_METHOD, metric=DISTANCE_METRIC)
    dist_main = squareform(pdist(x_main, metric=DISTANCE_METRIC))

    score_rows = []
    print(f"\n=== k sweep (top {N_MAIN} genes) ===")
    for k in K_RANGE:
        labels = cut_tree(z_main, k)
        sil = silhouette_score(dist_main, labels, metric="precomputed")
        col = f"{METHOD}_g{N_MAIN}_k{k}"
        assignments[col] = labels
        summ = summarize_clusters(assignments[col], groups, k, N_MAIN)
        summaries.append(summ)
        score_rows.append({"k": k, "silhouette": sil,
                           "min_cluster_size": summ["n_samples"].min()})
        print(f"\nk={k}  silhouette={sil:.3f}")
        print(summ[["cluster", "n_samples", "n_tumor", "n_normal"]]
              .to_string(index=False))
        report_small_clusters(assignments[col], groups)

    scores = pd.DataFrame(score_rows)
    plot_k_selection(scores, FIG_DIR / "08_hclust_k_selection.png")
    plot_membership(
        pd.concat(summaries), FIG_DIR / "08_hclust_membership_by_k.png",
        title=f"Hierarchical clustering composition by k "
              f"(top {N_MAIN:,} genes)",
    )
    plot_dendrogram(z_main, assignments.index, groups,
                    FIG_DIR / "08_hclust_dendrogram.png")

    # --- gene-number sweep at the primary k ----------------------------------
    print(f"\n=== Gene-number sweep (k={PRIMARY_K}) ===")
    for n_genes in GENE_COUNTS:
        col = f"{METHOD}_g{n_genes}_k{PRIMARY_K}"
        if col not in assignments:
            x = build_feature_matrix(log_expr, n_genes).to_numpy()
            z = linkage(x, method=LINKAGE_METHOD, metric=DISTANCE_METRIC)
            assignments[col] = cut_tree(z, PRIMARY_K)
            summaries.append(summarize_clusters(assignments[col], groups,
                                                PRIMARY_K, n_genes))
        tab = pd.crosstab(assignments[col], groups)
        print(f"\n{col}\n{tab.to_string()}")
        report_small_clusters(assignments[col], groups)

    out = TBL_DIR / f"{METHOD}_assignments.csv"
    assignments.to_csv(out)
    print(f"\nSaved assignments ({assignments.shape[1]} versions) -> {out}")

    summary = pd.concat(summaries, ignore_index=True)
    out = TBL_DIR / f"{METHOD}_k_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved cluster summary -> {out}")


if __name__ == "__main__":
    main()
