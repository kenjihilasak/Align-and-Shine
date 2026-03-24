# Paper Repository Guide

This repository currently behaves like a research workspace. For the paper submission, the cleanest strategy is to publish a separate official repository focused on the corpus artifact.

## Recommendation

Use two repositories:

- `align-and-shine`
  The frozen, paper-facing repository.
- `align-and-shine-dev` or `align-and-shine-lab`
  The ongoing research and development repository.

This split makes the paper artifact easier to understand, cite, review, and maintain. It also lets you keep adding work such as word alignment without making the official corpus release look unstable.

## What To Include In `align-and-shine`

- the paper-oriented `README.md`
- `LICENSE`
- `CITATION.cff`
- `config/`
- core `src/` scripts needed to reproduce the corpus
- the SentAlign variants used in the paper
- `data/<lang>/raw/`, `data/<lang>/wiki/`, `data/<lang>/viki/`
- `data/<lang>/aligned/` as the stable public release location
- a small amount of metadata and summary statistics

## What To Keep Out Of The Official Paper Repo

- exploratory notebooks that are not needed for reproduction
- broad `reports/` directories with many intermediate outputs
- pilot word-alignment experiments
- annotation workbooks unrelated to the sentence-aligned corpus release
- large temporary outputs under `runs/` unless they are essential

## Data Presentation

For readers of the paper, the most important thing is to find the final corpus immediately.

Recommended rule:

- final sentence pairs belong in `data/<lang>/aligned/`
- intermediate system outputs belong in `runs/`

If the raw data or full intermediate artifacts are too large for GitHub, keep the repository structure and publish the heavy files through Zenodo, OSF, Hugging Face, or GitHub Releases. The repo can then link to the archived files while still exposing metadata, manifests, and sample files.

## Why `runs/` Is Optional

The current scripts use `runs/<RUN_TAG>/...` as an internal execution layout. That is good for reproducibility, but paper readers should not need to inspect run folders just to locate the published corpus.

For the official release:

- keep `runs/` only if you want to document intermediate artifacts
- do not make `runs/` the only place where the final aligned pairs live

## Minimum Release Checklist

- `README.md` explains the task, languages, and file layout
- `LICENSE` is present
- `CITATION.cff` is present
- the final corpus is easy to find
- file formats are documented
- language coverage is explicit
- thresholds and model choices from the paper are versioned
- any external archive link is stable and public
