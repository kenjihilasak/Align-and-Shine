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
- `aligned/`
  Recommended release location for the final sentence-aligned corpus.

Suggested release convention for `aligned/`:

- one stable file per language, such as `align_and_shine_<lang>.tsv`
- a documented schema
- optional metadata or manifest files

This repository currently tracks only the directory skeleton so the structure is ready before large files are added manually.
