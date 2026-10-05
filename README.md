# HCC RNA-seq Classification

## Overview

This project investigates gene expression differences between hepatocellular
carcinoma (HCC) tumor tissue and matched non-tumor liver tissue using RNA-seq
data, and explores whether expression profiles can distinguish the two groups.

**Scientific question:** Can gene expression profiles distinguish hepatocellular
carcinoma (HCC) tumor tissue from matched non-tumor liver tissue?

## Dataset

**RNA-seq of 50 paired hepatocellular carcinoma samples** — refine.bio accession
**SRP068976** (50 tumor + 50 matched normal liver = 100 samples).

Source: https://www.refine.bio/experiments/SRP068976/rna-seq-of-50-paired-hepatocellular-carcinoma

Downloaded from refine.bio with quantile normalization skipped (un-normalized
tximport count estimates), so the data are suitable as input for DESeq2.

## Repository structure

```
.
├── data/SRP068976/          # expression matrix + sample metadata (from refine.bio)
├── scripts/                 # analysis pipeline (run in numeric order)
├── results/
│   ├── assn2/               # Assignment 2 outputs (steps 1–6)
│   │   ├── figures/         # generated figures (PNG)
│   │   └── tables/          # generated result tables (CSV)
│   └── assn3/               # Assignment 3 outputs (steps 7–10)
│       ├── figures/
│       └── tables/
└── assignment/              # written report (PDF)
```

### scripts/

| File | What it does |
|------|--------------|
| `01_load_annotate.py` | Load the expression matrix, map Ensembl IDs to gene symbols (mygene), and plot per-gene median expression. |
| `02_dimreduction.py` | PCA, t-SNE and UMAP of the samples, colored by tumor / normal. |
| `03_diffexp.py` | Differential expression (Tumor vs Normal) with PyDESeq2; volcano plot and top-50 table. |
| `04_heatmap.py` | Heatmap of the significant DEGs with a group side bar. |
| `05_enrichment.py` | GO:BP enrichment with two methods (g:Profiler over-representation and a Wilcoxon rank-sum gene-set test). |
| `06_combine_enrichment.py` | Merge the two enrichment results into a combined table and extract the top 10 shared terms. |
| `07_kmeans.py` | K-means clustering (top 5,000 genes, k = 2..6, elbow + silhouette), then k = 2 on 10 / 100 / 1,000 / 5,000 / 10,000 genes. |
| `08_hclust.py` | Hierarchical clustering (average linkage, 1 − Pearson r), same k and gene-number sweeps, plus a sample dendrogram. |
| `09_cluster_heatmap.py` | Heatmap of the top 5,000 genes with row/column dendrograms and K-means, hierarchical and Tumor/Normal annotation bars. |
| `10_chisq_tests.py` | Chi-squared tests between clustering versions and against Tumor/Normal, with Cramér's V, ARI and BH-adjusted p-values. |

Steps 1–6 make up Assignment 2 (exploration, differential expression,
enrichment); steps 7–10 make up Assignment 3 (unsupervised clustering).

### results/

Outputs are split by assignment; each folder has `figures/` (PNG) and `tables/` (CSV).

- `assn2/` — density plot, PCA/t-SNE/UMAP, volcano plot and DEG heatmap; sample sheet, gene annotation, differential expression results, significant DEG list and enrichment tables.
- `assn3/` — k-selection plots, cluster composition plots, sample dendrogram and clustering heatmap; cluster assignments and summaries for both methods, and the chi-squared test table.

The Assignment 3 scripts read `results/assn2/tables/sample_sheet.csv`, so step 1 must be run first.

## Requirements

Python 3.13. Pinned packages are listed in `requirements.txt`. To set up a
virtual environment on Windows:

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run the scripts in numeric order, e.g. `.venv\Scripts\python scripts\01_load_annotate.py`.
Steps 1 and 5 query online services (mygene, g:Profiler, Enrichr) and need
network access.

