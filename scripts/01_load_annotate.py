"""
Load the expression matrix and metadata, map Ensembl IDs to gene symbols,
and plot the distribution of per-gene median expression.

The mapping is kept as a side table; the matrix stays on Ensembl IDs for DESeq2.
Writes gene_annotation.csv, sample_sheet.csv and 01_density_median.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SRP068976"
EXPR_PATH = DATA_DIR / "SRP068976.tsv"
META_PATH = DATA_DIR / "metadata_SRP068976.tsv"
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# group colors reused across all figures
GROUP_COLORS = {"Tumor": "#d62728", "Normal": "#1f77b4"}


def load_expression() -> pd.DataFrame:
    """Load the expression matrix as genes (rows) x samples (columns)."""
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    expr.index.name = "ensembl_id"
    return expr


def load_sample_sheet(sample_ids) -> pd.DataFrame:
    """Build a sample -> group table from the metadata titles.

    Titles look like "Patient 1 Normal" / "Patient 1 Tumor"; the group is the
    last word of the title.
    """
    meta = pd.read_csv(META_PATH, sep="\t")
    meta = meta.set_index("refinebio_accession_code")
    group = (
        meta["refinebio_title"]
        .str.strip()
        .str.split()
        .str[-1]
        .str.capitalize()
    )
    sheet = pd.DataFrame({"group": group}).loc[list(sample_ids)]
    assert set(sheet["group"]) <= {"Tumor", "Normal"}, "Unexpected group labels"
    return sheet


def annotate_gene_symbols(ensembl_ids) -> pd.DataFrame:
    """Map Ensembl gene IDs to HGNC symbols using mygene."""
    import mygene

    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        list(ensembl_ids),
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        df_index=True,
        verbose=False,
    )
    # Keep the first symbol per Ensembl ID and drop duplicate query hits.
    res = res[~res.index.duplicated(keep="first")]
    annotation = pd.DataFrame(index=pd.Index(ensembl_ids, name="ensembl_id"))
    annotation["gene_symbol"] = res.reindex(annotation.index)["symbol"]
    return annotation


def main() -> None:
    expr = load_expression()
    sample_sheet = load_sample_sheet(expr.columns)
    sample_sheet.to_csv(TBL_DIR / "sample_sheet.csv")

    n_genes, n_samples = expr.shape

    annotation = annotate_gene_symbols(expr.index)
    annotation.to_csv(TBL_DIR / "gene_annotation.csv")
    n_mapped = annotation["gene_symbol"].notna().sum()

    print("=== Step 1: data summary ===")
    print(f"Expression matrix shape : {n_genes} genes x {n_samples} samples")
    print(f"Samples per group       : "
          f"{sample_sheet['group'].value_counts().to_dict()}")
    print(f"Genes with a symbol      : {n_mapped} / {n_genes} "
          f"({100 * n_mapped / n_genes:.1f}%)")

    # counts are un-normalized tximport estimates; log2(x+1) tames the dynamic
    # range before we look at per-gene statistics
    log_expr = np.log2(expr + 1.0)
    per_gene_median = log_expr.median(axis=1)

    q = per_gene_median.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    print("\n=== Per-gene median expression (log2) distribution ===")
    print(f"min={q[0.0]:.2f}  Q1={q[0.25]:.2f}  median={q[0.5]:.2f}  "
          f"Q3={q[0.75]:.2f}  max={q[1.0]:.2f}")
    n_silent = int((per_gene_median == 0).sum())
    print(f"Genes with median log2 expression == 0 (silent): {n_silent}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(per_gene_median, fill=True, color="#4c72b0", ax=ax)
    ax.set_xlabel("Per-gene median expression, log2(count + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-gene median expression (SRP068976)")
    fig.tight_layout()
    out = FIG_DIR / "01_density_median.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved density plot -> {out}")


if __name__ == "__main__":
    main()
