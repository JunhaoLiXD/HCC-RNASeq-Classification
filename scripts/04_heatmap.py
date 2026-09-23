"""
Step 4 - Heatmap of significantly differentially expressed genes.

What this script does:
  1. Load the full DE results from Step 3 and the expression matrix.
  2. Define the strict significant DEG set: padj < 0.05 AND |log2FC| > 2.
     Save that gene list (it is also the input for Step 5 enrichment).
  3. For readability, take the top 25 up- and top 25 down-regulated genes by
     adjusted p-value (50 genes) and draw a z-scored clustermap with a top side
     bar colored by sample group (Tumor / Normal).

Note: the strict set has 1,678 genes; showing all of them with labels is not
readable, so the heatmap displays the 50 most significant. The full significant
list is saved to results/tables/.

Outputs:
  results/tables/significant_degs.csv   strict DEG list (padj<0.05 & |log2FC|>2)
  results/figures/04_heatmap.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Project paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}

# Strict significance thresholds.
ALPHA = 0.05
LFC_THRESHOLD = 2.0
N_PER_SIDE = 25          # top up- and down-regulated genes shown in the heatmap


def main() -> None:
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(TBL_DIR / "sample_sheet.csv", index_col=0)
    sample_sheet = sample_sheet.loc[expr.columns]
    res = pd.read_csv(TBL_DIR / "diffexp_full.csv", index_col=0)

    # 1. Strict significant DEG set ------------------------------------------
    sig = res.dropna(subset=["padj", "log2FoldChange"])
    sig = sig[(sig["padj"] < ALPHA) & (sig["log2FoldChange"].abs() > LFC_THRESHOLD)]
    sig.to_csv(TBL_DIR / "significant_degs.csv")
    print(f"Strict significant DEGs (padj<{ALPHA} & |log2FC|>{LFC_THRESHOLD}): "
          f"{len(sig)}")

    # 2. Top genes for the heatmap (balanced up/down) ------------------------
    up = sig[sig["log2FoldChange"] > 0].nsmallest(N_PER_SIDE, "padj")
    down = sig[sig["log2FoldChange"] < 0].nsmallest(N_PER_SIDE, "padj")
    selected = pd.concat([up, down])
    print(f"Genes shown in heatmap: {len(selected)} "
          f"({len(up)} up, {len(down)} down)")

    # 3. Build the z-scored expression matrix (genes x samples) --------------
    log_expr = np.log2(expr.loc[selected.index] + 1.0)
    labels = selected["gene_symbol"].fillna(selected.index.to_series())
    log_expr.index = labels.to_numpy()

    # Column color strip encoding the sample group.
    col_colors = sample_sheet["group"].map(GROUP_COLORS)
    col_colors.name = "Group"

    g = sns.clustermap(
        log_expr,
        z_score=0,                     # standardize each gene across samples
        cmap="RdBu_r",
        center=0,
        col_colors=col_colors.to_numpy(),
        xticklabels=False,             # 100 sample names would be unreadable
        yticklabels=True,
        figsize=(11, 12),
        cbar_kws={"label": "Row z-score"},
        dendrogram_ratio=(0.1, 0.12),
    )
    g.ax_heatmap.set_xlabel("Samples (n=100)")
    g.ax_heatmap.set_ylabel("Top 50 differentially expressed genes")

    # Legend for the group side bar.
    handles = [mpatches.Patch(color=c, label=grp) for grp, c in GROUP_COLORS.items()]
    g.ax_heatmap.legend(
        handles=handles, title="Group",
        bbox_to_anchor=(1.25, 1.05), loc="upper left", frameon=True,
    )
    g.figure.suptitle(
        "Top 50 significant DEGs (padj<0.05 & |log2FC|>2)", y=1.02, fontsize=13
    )

    out = FIG_DIR / "04_heatmap.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved heatmap -> {out}")


if __name__ == "__main__":
    main()
