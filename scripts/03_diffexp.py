"""
Step 3 - Differential expression analysis (Tumor vs Normal) with PyDESeq2.

What this script does:
  1. Load the expression matrix and the sample -> group sheet from Step 1.
  2. Filter low-count genes and round the tximport count estimates to integers,
     as DESeq2 expects raw-like counts (design = ~condition, two-group test).
  3. Run PyDESeq2 with Normal as the reference so log2FoldChange is Tumor / Normal.
  4. Save the full results table, save the top 50 differentially expressed genes,
     and draw a volcano plot with significance thresholds and a legend.

Design: ~condition (simple two-group comparison, no patient blocking) to match
the assignment's referenced refinebio-examples tutorial.

Outputs:
  results/tables/diffexp_full.csv       all genes with log2FC, p, padj
  results/tables/diffexp_top50.csv      top 50 genes by adjusted p-value
  results/figures/03_volcano.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# --- Project paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"

GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}

# Analysis parameters.
MIN_COUNT = 10          # a gene must exceed this count ...
MIN_SAMPLES = 10        # ... in at least this many samples to be kept
ALPHA = 0.05            # adjusted p-value significance threshold
LFC_THRESHOLD = 1.0     # |log2FC| threshold used for the volcano coloring
N_TOP = 50              # number of top genes to tabulate / label


def load_inputs():
    """Load expression, sample sheet, and gene-symbol annotation."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    sample_sheet = pd.read_csv(TBL_DIR / "sample_sheet.csv", index_col=0)
    sample_sheet = sample_sheet.loc[expr.columns]
    annotation = pd.read_csv(TBL_DIR / "gene_annotation.csv", index_col=0)
    return expr, sample_sheet, annotation


def run_deseq(expr: pd.DataFrame, sample_sheet: pd.DataFrame) -> pd.DataFrame:
    """Run PyDESeq2 and return the Tumor-vs-Normal results table."""
    # Low-count filter, then round to integer counts for DESeq2.
    keep = (expr > MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    counts = expr.loc[keep].round().astype(int)
    print(f"Genes kept after low-count filter: {int(keep.sum())}")

    # PyDESeq2 wants samples as rows for both counts and metadata.
    counts_t = counts.T
    metadata = sample_sheet.rename(columns={"group": "condition"})

    dds = DeseqDataSet(
        counts=counts_t,
        metadata=metadata,
        design="~condition",
        ref_level=["condition", "Normal"],
        quiet=True,
    )
    dds.deseq2()

    stats = DeseqStats(
        dds, contrast=["condition", "Tumor", "Normal"], alpha=ALPHA, quiet=True
    )
    stats.summary()
    return stats.results_df


def volcano_plot(res: pd.DataFrame, out_path: Path) -> None:
    """Volcano plot colored by up / down / not-significant."""
    df = res.dropna(subset=["padj", "log2FoldChange"]).copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    sig = df["padj"] < ALPHA
    up = sig & (df["log2FoldChange"] >= LFC_THRESHOLD)
    down = sig & (df["log2FoldChange"] <= -LFC_THRESHOLD)
    ns = ~(up | down)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df.loc[ns, "log2FoldChange"], df.loc[ns, "neg_log10_padj"],
               c="lightgrey", s=8, label="Not significant")
    ax.scatter(df.loc[up, "log2FoldChange"], df.loc[up, "neg_log10_padj"],
               c=GROUP_COLORS["Tumor"], s=10, label="Up in Tumor")
    ax.scatter(df.loc[down, "log2FoldChange"], df.loc[down, "neg_log10_padj"],
               c=GROUP_COLORS["Normal"], s=10, label="Down in Tumor")

    # Threshold guide lines.
    ax.axhline(-np.log10(ALPHA), color="black", ls="--", lw=0.7)
    ax.axvline(LFC_THRESHOLD, color="black", ls="--", lw=0.7)
    ax.axvline(-LFC_THRESHOLD, color="black", ls="--", lw=0.7)

    # Label the 15 most significant genes by symbol.
    top = df.nsmallest(15, "padj")
    texts = []
    for _, row in top.iterrows():
        label = row["gene_symbol"] if isinstance(row["gene_symbol"], str) else row.name
        texts.append(ax.text(row["log2FoldChange"], row["neg_log10_padj"],
                             label, fontsize=7))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))

    ax.set_xlabel("log2 fold change (Tumor / Normal)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("Differential expression: Tumor vs Normal")
    ax.legend(title="Gene status", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved volcano -> {out_path}")


def main() -> None:
    expr, sample_sheet, annotation = load_inputs()

    res = run_deseq(expr, sample_sheet)
    res = res.join(annotation["gene_symbol"])
    res.index.name = "ensembl_id"
    res = res.sort_values("padj")

    # Full results table.
    full_path = TBL_DIR / "diffexp_full.csv"
    res.to_csv(full_path)
    print(f"Saved full results -> {full_path}")

    # Significance counts.
    sig = res["padj"] < ALPHA
    up = sig & (res["log2FoldChange"] >= LFC_THRESHOLD)
    down = sig & (res["log2FoldChange"] <= -LFC_THRESHOLD)
    print(f"Significant genes (padj < {ALPHA}): {int(sig.sum())}")
    print(f"  Up in Tumor   (log2FC >= {LFC_THRESHOLD}): {int(up.sum())}")
    print(f"  Down in Tumor (log2FC <= -{LFC_THRESHOLD}): {int(down.sum())}")

    # Top 50 by adjusted p-value.
    cols = ["gene_symbol", "baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]
    top50 = res[cols].head(N_TOP)
    top50_path = TBL_DIR / "diffexp_top50.csv"
    top50.to_csv(top50_path)
    print(f"Saved top {N_TOP} -> {top50_path}")

    # Volcano plot.
    volcano_plot(res, FIG_DIR / "03_volcano.png")


if __name__ == "__main__":
    main()
