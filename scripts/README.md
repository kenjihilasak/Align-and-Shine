# Scripts

This directory groups scripts that are useful for reproduction but are not part of the main top-level paper interface.

## Structure

- `slurm/`
  Example launchers for HPC and SLURM environments.
- `baselines/hunalign/`
  Helper files for classic Hunalign-style baselines.
- `legacy/`
  Older or ad hoc utilities kept for traceability, not required for the main paper pipeline.

## Recommended Reading Order

If you are mainly interested in the paper artifact, start with:

1. `src/prepare_sentalign_data.py`
2. `src/filter_full_corpus_by_best_thresholds.py`
3. `config/best_thresholds_strict_fullcorpus.csv`
4. `data/<lang>/aligned/`

Use the scripts under this folder only if you need cluster launchers, baseline comparisons, or older utilities.
