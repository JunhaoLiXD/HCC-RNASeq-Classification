"""
Step 6 & 7 - Combine the enrichment results and extract the top terms.

Step 6: build a joint table of the two GO:BP enrichment methods. Each unique GO
term is one row holding each method's adjusted p-value, plus:
  n_methods_tested      - how many methods included the term in their analysis
  n_methods_significant - how many methods called the term significant (padj<0.05)

Step 7: from the combined table, take the top 10 terms enriched in all/most
methods (ranked by number of significant methods, then by combined significance).

Caveat: g:Profiler only returns significant terms, so a term's presence in the
gProfiler output is treated as "tested and significant". The Wilcoxon method
returns every tested term with its padj.

Outputs:
  results/tables/enrichment_combined.csv
  results/tables/enrichment_top10.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL_DIR = ROOT / "results" / "tables"

ALPHA = 0.05


def main() -> None:
    gp = pd.read_csv(TBL_DIR / "enrichment_gprofiler_GO.csv")
    wx = pd.read_csv(TBL_DIR / "enrichment_wilcoxon_GO.csv")

    # gProfiler output = significant terms only.
    gp = gp[["term_id", "term_name", "padj"]].rename(
        columns={"padj": "gprofiler_padj"})
    gp["gprofiler_tested"] = True
    gp["gprofiler_significant"] = gp["gprofiler_padj"] < ALPHA

    # Wilcoxon output = every tested term.
    wx = wx[["term_id", "term_name", "padj"]].rename(
        columns={"padj": "wilcoxon_padj"})
    wx["wilcoxon_tested"] = True
    wx["wilcoxon_significant"] = wx["wilcoxon_padj"] < ALPHA

    # Union on GO id; prefer the (cleaner) Wilcoxon term names when present.
    combined = pd.merge(
        wx, gp, on="term_id", how="outer", suffixes=("_wx", "_gp"))
    combined["term_name"] = combined["term_name_wx"].fillna(
        combined["term_name_gp"])
    combined = combined.drop(columns=["term_name_wx", "term_name_gp"])

    for col in ["gprofiler_tested", "gprofiler_significant",
                "wilcoxon_tested", "wilcoxon_significant"]:
        combined[col] = combined[col].fillna(False)

    combined["n_methods_tested"] = (
        combined["gprofiler_tested"].astype(int)
        + combined["wilcoxon_tested"].astype(int))
    combined["n_methods_significant"] = (
        combined["gprofiler_significant"].astype(int)
        + combined["wilcoxon_significant"].astype(int))

    # Combined significance score for ranking: sum of -log10(padj) over methods.
    def neglog(p):
        return -np.log10(p.clip(lower=1e-300)) if p is not None else 0.0

    combined["score"] = (
        neglog(combined["gprofiler_padj"].fillna(1.0))
        + neglog(combined["wilcoxon_padj"].fillna(1.0)))

    cols = ["term_id", "term_name", "gprofiler_padj", "wilcoxon_padj",
            "n_methods_tested", "n_methods_significant", "score"]
    combined = combined[cols].sort_values(
        ["n_methods_significant", "score"], ascending=[False, False]
    ).reset_index(drop=True)

    out = TBL_DIR / "enrichment_combined.csv"
    combined.to_csv(out, index=False)
    print(f"Combined table: {len(combined)} unique GO terms -> {out}")
    print("Terms significant in both methods : "
          f"{int((combined['n_methods_significant'] == 2).sum())}")
    print("Terms significant in one method   : "
          f"{int((combined['n_methods_significant'] == 1).sum())}")

    # Step 7: top 10 enriched in all/most methods.
    top10 = combined.head(10).drop(columns=["score"])
    top10_out = TBL_DIR / "enrichment_top10.csv"
    top10.to_csv(top10_out, index=False)
    print(f"\nTop 10 shared enriched terms -> {top10_out}")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
