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
│   ├── figures/             # generated figures (PNG)
│   └── tables/              # generated result tables (CSV)
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

### results/

- `figures/` — one PNG per analysis step (density plot, PCA/t-SNE/UMAP, volcano, heatmap).
- `tables/` — differential expression results, significant DEG list, and enrichment tables.

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

