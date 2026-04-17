# Data Layout

This directory contains the multilingual corpus structure used by the paper.

Language folders:

- `ca`
- `en`
- `es`
- `fr`
- `it`

Expected subdirectories per language:

- `raw/`
  Raw source files, typically `.jsonl` exports from Wikipedia and Vikidia.
- `wiki/`
  One `.txt` file per document, one sentence per line, Wikipedia side.
- `viki/`
  One `.txt` file per document, one sentence per line, Vikidia side.
- `alignments/`
  Document-level alignment files in `.path` format.
- `aligned/`
  Final released aligned outputs, including clean per-language CSV exports.
