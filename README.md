# Align and Shine

Official repository for the paper *Align and Shine: Building high-quality sentence-aligned corpora for multilingual text simplification*.

This repository is the paper-facing release. It focuses on the multilingual sentence-aligned corpus, the core alignment pipeline, and the configuration needed to reproduce the main corpus-building steps.

The development repository lives separately at:

- https://github.com/kenjihilasak/Align-and-Shine-dev

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
|- docs/
|- runs/
|- scripts/
|  |- slurm/
|  |- baselines/
|  `- legacy/
|- src/
|- SentAlign_BGE/
|- SentAlign_laBSE/
`- SentAlign_SONAR/
```

The top level is intentionally kept focused on the corpus, configuration, and core pipeline. Cluster launchers and baseline-specific helpers live under `scripts/` so they do not distract from the main paper artifact.

## What Is Included

- the corpus directory skeleton under `data/`
- the threshold configuration used in the paper
- the core preparation and filtering scripts
- the three SentAlign variants used in the paper
- optional SLURM launchers grouped under `scripts/slurm/`
- optional Hunalign baseline helpers grouped under `scripts/baselines/hunalign/`

## What Is Intentionally Secondary

This paper repository does not center the broader experimental layer from the development repo, such as:

- word-alignment experiments
- exploratory notebooks
- large report dumps
- pilot annotation assets not required for the paper artifact

Some historical or environment-specific helpers are still preserved under `scripts/legacy/`, but they are not part of the main paper pipeline.

## Data Layout

Per language, the expected directories are:

- `raw/`: source `.jsonl` files
- `wiki/`: one Wikipedia document per `.txt`, one sentence per line
- `viki/`: one Vikidia document per `.txt`, one sentence per line
- `aligned/`: recommended location for the final released sentence pairs

The final corpus should be easy to find in `data/<lang>/aligned/`. Intermediate system outputs can stay in `runs/`.

## Licensing

This repository separates code licensing from data licensing:

- Code: `Apache-2.0` in `LICENSE`
- Corpus/data policy: `DATA_LICENSE.md`
- Upstream attribution guidance: `ATTRIBUTION.md`

This split is intentional because the repository code and the text-derived corpus do not have the same licensing context.

## Core Pipeline

1. Prepare sentence-per-line document files with `src/prepare_sentalign_data.py`
2. Run full-corpus alignment with the paper models
3. Apply versioned thresholds with `src/filter_full_corpus_by_best_thresholds.py`
4. Optionally analyze filtered outputs with `src/analyze_filtered_pairs_stats.py`

If you use an HPC cluster, example launchers are available in `scripts/slurm/`. They are provided as environment-specific templates, not as the main public interface of the repository.

## Before Public Release

Still recommended before publishing this repository:

- add `CITATION.cff`
- add the final aligned corpus files under `data/<lang>/aligned/`
- add any archive link if large files are hosted outside GitHub
