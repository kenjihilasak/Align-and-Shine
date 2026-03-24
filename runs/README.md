# Runs

This directory is reserved for intermediate run artifacts such as:

- raw alignment outputs
- filtered alignment outputs
- per-run metadata
- execution logs copied from cluster jobs

The current code expects layouts such as:

`runs/<RUN_TAG>/<lang>/01_raw_alignments/<model>/`

and

`runs/<RUN_TAG>/<lang>/02_filtered_alignments/<model>/`

For the official paper release, `runs/` is optional. The final released corpus should be exposed in `data/<lang>/aligned/` so users do not need to browse intermediate execution folders.
