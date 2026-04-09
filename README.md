# Align and Shine

Official repository for the paper *Align and Shine: Building high-quality sentence-aligned corpora for multilingual text simplification*.

This repository provides the released multilingual sentence-aligned corpus, the core alignment pipeline, and the configuration used to build the corpus reported in the paper.

## Languages

The released corpus structure is organized for:

- `ca`
- `en`
- `es`
- `fr`
- `it`

## Repository Layout

```text
Align-and-Shine/
|- README.md
|- config/
|- data/
|  |- <lang>/
|  |  |- raw/
|  |  |- wiki/
|  |  |- viki/
|  |  `- aligned/
|- runs/
|- scripts/
|  |- slurm/
|  `- legacy/
|- src/
|- SentAlign_BGE/
|- SentAlign_laBSE/
`- SentAlign_SONAR/
```

The top level is kept focused on the released corpus, configuration, and the core pipeline components used to prepare and filter alignments.

## Repository Contents

- the corpus directory skeleton under `data/`
- the released final sentence alignments under `data/<lang>/aligned/`
- the threshold configuration used in the paper
- the core preparation and filtering scripts
- the three SentAlign variants used in the paper
- optional SLURM launchers grouped under `scripts/slurm/`
- legacy helper scripts grouped under `scripts/legacy/`

Large raw source files are tracked with Git LFS. If you need the full raw inputs, install Git LFS before cloning or run `git lfs pull` after cloning the repository.

## Data Layout

Per language, the expected directories are:

- `raw/`: source `.jsonl` and related input files
- `wiki/`: one Wikipedia document per `.txt`, one sentence per line
- `viki/`: one Vikidia document per `.txt`, one sentence per line
- `aligned/`: final sentence-aligned corpus released with this repository

The final corpus is available in `data/<lang>/aligned/`. Intermediate system outputs and run-specific artifacts are stored under `runs/`.

## Licensing

This repository separates code licensing from data licensing:

- Code: `Apache-2.0` in `LICENSE`
- Corpus/data policy: `DATA_LICENSE.md`
- Upstream attribution guidance: `ATTRIBUTION.md`

## Core Pipeline

1. Prepare sentence-per-line document files with `src/prepare_sentalign_data.py`
2. Run full-corpus alignment with the paper models
3. Apply versioned thresholds with `src/filter_full_corpus_by_best_thresholds.py`
4. Optionally analyze filtered outputs with `src/analyze_filtered_pairs_stats.py`

If you use an HPC cluster, example launchers are available in `scripts/slurm/`.

## Acknowledgments

This repository is part of a project that has received funding from the European Union’s Horizon Europe research and innovation program under Grant Agreement No. 101132431 (iDEM Project). The University of Leeds was funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee (Grant Agreement No. 10103529). The views and opinions expressed in this document are solely those of the author(s) and do not necessarily reflect the views of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
