# Data Licensing

This repository separates code licensing from data licensing.

- Code and repository scaffolding: see `LICENSE` (`Apache-2.0`)
- Data and text-derived artifacts: see this file and `ATTRIBUTION.md`

## Intended Scope

This file applies to the corpus content stored or later released under:

- `data/<lang>/aligned/`
- any metadata files distributed specifically as part of the released aligned corpus

It does not override the upstream terms that apply to copied or lightly transformed source text under:

- `data/<lang>/raw/`
- `data/<lang>/wiki/`
- `data/<lang>/viki/`

## Recommended Release Terms For The Aligned Corpus

The intended public release terms for the final aligned corpus are:

- `CC BY-SA 4.0`

Rationale:

- Wikipedia text on most Wikimedia projects is generally reusable under `CC BY-SA 4.0`, with GFDL also relevant in some cases.
- Vikidia text is available under `CC BY-SA 3.0` and GFDL.
- The aligned corpus is a text-derived research artifact that should remain open, attributable, and share-alike.

Because the corpus is derived from upstream collaborative sources with their own licensing and attribution requirements, reusers should also review `ATTRIBUTION.md`.

## Important Upstream Note

The final corpus release should be treated as a derivative research dataset built from upstream text sources. Reusers must preserve:

- attribution to the upstream projects
- links to the applicable upstream licenses
- indication that the corpus contains aligned and processed text derived from Wikipedia and Vikidia
- share-alike obligations where applicable

## Caution

This file documents the intended repository policy for the corpus release. It is not legal advice. If your institution requires formal legal review for data publication, use that review before publishing the final corpus files.
