# Scripts

This directory groups supporting scripts for running and reproducing the alignment pipeline.

## Structure

- `slurm/`
  Example launchers for HPC and SLURM environments.
- `legacy/`
  Older or environment-specific helpers kept for reference.

## Recommended Entry Points

If you are mainly interested in the released corpus and its construction, start with:

1. `src/prepare_sentalign_data.py`
2. `src/filter_full_corpus_by_best_thresholds.py`
3. `config/best_thresholds_strict_fullcorpus.csv`
4. `data/<lang>/aligned/`

Use the scripts in this directory when you need cluster launchers or compatibility utilities around the main pipeline.
