"""
Step 2 - Dimensionality reduction: PCA, t-SNE, and UMAP.

What this script does:
  1. Load the expression matrix and the sample -> group sheet from Step 1.
  2. Filter out low-count genes, log2-transform, keep the top high-variance
     genes, and z-score each gene so no single gene dominates.
  3. Project the 100 samples into 2D with PCA, t-SNE, and UMAP.
  4. Draw one scatter plot per method, colored by Tumor / Normal, with axis
     labels and a legend.

Outputs:
  results/figures/02_pca.png
  results/figures/02_tsne.png
  results/figures/02_umap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- Project paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}

# Analysis parameters (kept here so the summary can cite exact values).
MIN_COUNT = 10          # a gene must exceed this count ...
MIN_SAMPLES = 10        # ... in at least this many samples to be kept
N_TOP_VARIABLE = 2000   # number of most-variable genes fed to the projections
RANDOM_STATE = 42


def load_inputs():
    """Load the expression matrix and the sample sheet built in Step 1."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(TBL_DIR / "sample_sheet.csv", index_col=0)
    sample_sheet = sample_sheet.loc[expr.columns]
    return expr, sample_sheet


def build_feature_matrix(expr: pd.DataFrame) -> pd.DataFrame:
    """Return a samples x genes matrix of z-scored, high-variance log2 values."""
    # Drop genes that are lowly expressed across the cohort.
    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    filtered = expr.loc[keep]

    log_expr = np.log2(filtered + 1.0)

    # Keep the most variable genes; these carry the biological signal.
    top_genes = log_expr.var(axis=1).nlargest(N_TOP_VARIABLE).index
    log_expr = log_expr.loc[top_genes]

    # Samples become rows; z-score each gene (column) so scales are comparable.
    x = log_expr.T
    x_scaled = StandardScaler().fit_transform(x)
    print(f"Genes kept after low-count filter : {int(keep.sum())}")
    print(f"High-variance genes used          : {len(top_genes)}")
    return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)


def scatter_by_group(coords, groups, xlabel, ylabel, title, out_path):
    """Draw a 2D scatter colored by sample group."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for group, color in GROUP_COLORS.items():
        mask = (groups == group).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=group, s=45, alpha=0.8, edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    expr, sample_sheet = load_inputs()
    groups = sample_sheet["group"]

    features = build_feature_matrix(expr)
    x = features.to_numpy()

    # --- PCA -----------------------------------------------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(x)
    var = pca.explained_variance_ratio_ * 100
    scatter_by_group(
        pca_coords, groups,
        xlabel=f"PC1 ({var[0]:.1f}% variance)",
        ylabel=f"PC2 ({var[1]:.1f}% variance)",
        title="PCA of SRP068976 samples",
        out_path=FIG_DIR / "02_pca.png",
    )
    print(f"PCA explained variance (PC1, PC2): {var[0]:.1f}%, {var[1]:.1f}%")

    # --- t-SNE ---------------------------------------------------------------
    tsne = TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", random_state=RANDOM_STATE,
    )
    tsne_coords = tsne.fit_transform(x)
    scatter_by_group(
        tsne_coords, groups,
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        title="t-SNE of SRP068976 samples (perplexity=30)",
        out_path=FIG_DIR / "02_tsne.png",
    )

    # --- UMAP ----------------------------------------------------------------
    import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    umap_coords = reducer.fit_transform(x)
    scatter_by_group(
        umap_coords, groups,
        xlabel="UMAP 1", ylabel="UMAP 2",
        title="UMAP of SRP068976 samples (n_neighbors=15)",
        out_path=FIG_DIR / "02_umap.png",
    )


if __name__ == "__main__":
    main()
