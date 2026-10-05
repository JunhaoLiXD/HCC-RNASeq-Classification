"""
Chi-squared tests comparing clustering results with each other and with the
Tumor / Normal groups, collected in one table.

Comparisons:
  1. Gene-number versions within each method (k = primary k; C(5,2) = 10 pairs
     per method).
  2. K-means vs hierarchical clustering at the same number of genes.
  3. Every clustering result (all k versions and all gene-number versions)
     vs the Tumor / Normal groups (test of independence).

Each row reports chi2, dof and p-value from scipy's chi2_contingency (default
Yates continuity correction, which only affects 2x2 tables), plus Cramer's V
(from the uncorrected chi2), the adjusted Rand index (ARI) and the smallest
expected cell count. All p-values are BH-adjusted together.

Writes chisq_tests.csv.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
TBL_DIR = ROOT / "results" / "assn3" / "tables"
ASSN2_TBL_DIR = ROOT / "results" / "assn2" / "tables"   # sample_sheet.csv (Step 1)

METHODS = ["kmeans", "hclust"]
PRIMARY_K = 2
GENE_COUNTS = [10, 100, 1000, 5000, 10000]
ALPHA = 0.05


def chisq_compare(a: pd.Series, b: pd.Series) -> dict:
    """Chi-squared test of independence between two labelings."""
    table = pd.crosstab(a, b)
    out = {
        "n_clusters_a": table.shape[0], "n_clusters_b": table.shape[1],
        "ari": adjusted_rand_score(a, b),
    }
    if min(table.shape) < 2:
        # A single cluster leaves nothing to test.
        print(f"  WARNING: {a.name} vs {b.name} has a single category; "
              "test skipped")
        return {**out, "chi2": np.nan, "dof": 0, "pvalue": np.nan,
                "cramers_v": np.nan, "min_expected": np.nan}

    chi2, p, dof, expected = chi2_contingency(table)
    chi2_raw = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    v = np.sqrt(chi2_raw / (n * (min(table.shape) - 1)))
    return {**out, "chi2": chi2, "dof": dof, "pvalue": p, "cramers_v": v,
            "min_expected": expected.min()}


def main() -> None:
    sample_sheet = pd.read_csv(ASSN2_TBL_DIR / "sample_sheet.csv", index_col=0)
    assignments = pd.concat(
        [pd.read_csv(TBL_DIR / f"{m}_assignments.csv", index_col=0)
         for m in METHODS], axis=1)
    groups = sample_sheet.loc[assignments.index, "group"]

    rows = []

    def add(comparison_type, col_a, col_b, series_b):
        res = chisq_compare(assignments[col_a], series_b)
        rows.append({"comparison_type": comparison_type, "result_a": col_a,
                     "result_b": col_b, **res})

    # 1. Gene-number versions within each method.
    for method in METHODS:
        cols = [f"{method}_g{n}_k{PRIMARY_K}" for n in GENE_COUNTS]
        for col_a, col_b in combinations(cols, 2):
            add(f"gene_number_{method}", col_a, col_b, assignments[col_b])

    # 2. Between methods at the same number of genes.
    for n in GENE_COUNTS:
        col_a = f"kmeans_g{n}_k{PRIMARY_K}"
        col_b = f"hclust_g{n}_k{PRIMARY_K}"
        add("between_methods", col_a, col_b, assignments[col_b])

    # 3. Every clustering result vs the Tumor / Normal groups.
    for col in assignments.columns:
        add("vs_group", col, "group", groups)

    table = pd.DataFrame(rows)
    tested = table["pvalue"].notna()
    table["padj"] = np.nan
    table.loc[tested, "padj"] = multipletests(
        table.loc[tested, "pvalue"], method="fdr_bh")[1]

    cols = ["comparison_type", "result_a", "result_b", "n_clusters_a",
            "n_clusters_b", "chi2", "dof", "pvalue", "padj", "cramers_v",
            "ari", "min_expected"]
    table = table[cols]
    out = TBL_DIR / "chisq_tests.csv"
    table.to_csv(out, index=False)
    print(f"Saved {len(table)} tests -> {out}")

    print(f"\nSignificant after BH (padj < {ALPHA}): "
          f"{int((table['padj'] < ALPHA).sum())} / {int(tested.sum())}")
    print(f"Tests with a minimum expected count < 5: "
          f"{int((table['min_expected'] < 5).sum())}")

    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.3g}".format):
        print(table.drop(columns="comparison_type").to_string(index=False))


if __name__ == "__main__":
    main()
