"""
Gene-set enrichment by two methods, both over GO Biological Process so their
results can be combined in Step 6 (a term may be tested/significant by both).

  Method 1: g:Profiler over-representation + GO:BP
      Input = the strict significant DEG list (padj<0.05 & |log2FC|>2).

  Method 2: Wilcoxon rank-sum test + GO:BP
      Ranks all tested genes by the DESeq2 Wald statistic, then for each GO:BP
      set compares member vs non-member ranks with a Mann-Whitney U test.
      p-values are BH-corrected.

Writes enrichment_gprofiler_GO.csv and enrichment_wilcoxon_GO.csv.
"""

import re
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
TBL_DIR = ROOT / "results" / "tables"

MIN_SET_SIZE = 10
MAX_SET_SIZE = 500
ALPHA = 0.05
GO_LIBRARY = "GO_Biological_Process_2025"

_GO_ID = re.compile(r"\((GO:\d+)\)")


def _go_id_from_name(name: str) -> str:
    """Extract a GO id like GO:0000278 from an Enrichr term name."""
    m = _GO_ID.search(name)
    return m.group(1) if m else name


def method1_gprofiler_go() -> pd.DataFrame:
    """Over-representation of the DEG list in GO:BP via g:Profiler."""
    from gprofiler import GProfiler

    sig = pd.read_csv(TBL_DIR / "significant_degs.csv", index_col=0)
    query = list(sig.index)
    print(f"[gProfiler2] query genes: {len(query)}")

    gp = GProfiler(return_dataframe=True)
    res = gp.profile(
        organism="hsapiens",
        query=query,
        sources=["GO:BP"],
        user_threshold=0.05,
        significance_threshold_method="g_SCS",
        no_evidences=True,
    )
    res = res.rename(columns={
        "native": "term_id", "name": "term_name", "p_value": "padj",
        "intersection_size": "overlap",
    })
    res = res[["term_id", "term_name", "padj", "term_size", "overlap"]]
    res = res.sort_values("padj").reset_index(drop=True)
    out = TBL_DIR / "enrichment_gprofiler_GO.csv"
    res.to_csv(out, index=False)
    print(f"[gProfiler2] significant GO:BP terms: {len(res)} -> {out}")
    return res


def wilcoxon_geneset_test(library_name: str) -> pd.DataFrame:
    """Wilcoxon rank-sum gene-set test of a gene-set library on the DE ranking."""
    import gseapy

    de = pd.read_csv(TBL_DIR / "diffexp_full.csv", index_col=0)
    de = de.dropna(subset=["gene_symbol", "stat"])
    # If a symbol maps to several Ensembl IDs, keep the strongest statistic.
    de = de.reindex(de["stat"].abs().sort_values(ascending=False).index)
    de = de.drop_duplicates("gene_symbol")
    ranks = pd.Series(de["stat"].to_numpy(), index=de["gene_symbol"].to_numpy())
    universe = set(ranks.index)

    library = gseapy.get_library(name=library_name, organism="human")

    rows = []
    for term, genes in library.items():
        members = [g for g in genes if g in universe]
        if not (MIN_SET_SIZE <= len(members) <= MAX_SET_SIZE):
            continue
        in_set = ranks[members]
        out_set = ranks.drop(index=members)
        _, p = mannwhitneyu(in_set, out_set, alternative="two-sided")
        direction = "up" if in_set.median() > out_set.median() else "down"
        rows.append({
            "term_id": _go_id_from_name(term),
            "term_name": re.sub(r"\s*\(GO:\d+\)", "", term),
            "pvalue": p, "term_size": len(members), "overlap": len(members),
            "median_stat_in_set": in_set.median(), "direction": direction,
        })

    res = pd.DataFrame(rows)
    res["padj"] = multipletests(res["pvalue"], method="fdr_bh")[1]
    return res.sort_values("padj").reset_index(drop=True)


def main() -> None:
    print("=== Method 1: gProfiler2 + GO:BP ===")
    go1 = method1_gprofiler_go()
    print(go1.head(10).to_string(index=False))

    print("\n=== Method 2: Wilcoxon rank-sum + GO:BP ===")
    go2 = wilcoxon_geneset_test(GO_LIBRARY)
    out = TBL_DIR / "enrichment_wilcoxon_GO.csv"
    go2.to_csv(out, index=False)
    n_sig = int((go2["padj"] < ALPHA).sum())
    print(f"tested sets: {len(go2)}, significant (padj<{ALPHA}): {n_sig} -> {out}")
    print(go2[["term_name", "term_size", "direction", "pvalue", "padj"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
