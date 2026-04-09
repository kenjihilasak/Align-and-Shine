# Runs

This directory stores intermediate artifacts produced while building or re-running the corpus pipeline.

Typical contents include:

- raw alignment outputs
- filtered alignment outputs
- per-run metadata
- execution logs copied from cluster jobs

Typical layouts include:

`runs/<RUN_TAG>/<lang>/01_raw_alignments/<model>/`

and

`runs/<RUN_TAG>/<lang>/02_filtered_alignments/<model>/`

Most users do not need to work with this directory directly. The released sentence-aligned corpus is exposed in `data/<lang>/aligned/`.
