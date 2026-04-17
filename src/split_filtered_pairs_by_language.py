import argparse
import csv
import re
from pathlib import Path


DEFAULT_RUN_TAG = "2026-02-28_fullCorpus_v1"
DEFAULT_LANGS = "en,es,fr,it,ca"
DEFAULT_REPORTS_DIR = "reports/filtered_pairs_stats"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUT_SUBDIR = "aligned"
CSV_ENCODING = "utf-8-sig"
SUPPORTED_LANGUAGES = ("en", "es", "fr", "it", "ca")
MIN_WORDS_EXCLUSIVE = 3

REQUIRED_COLUMNS = {"language", "doc_name", "src_text", "tgt_text"}
DOC_NUMBER_RE = re.compile(r"doc-(\d+)\.txt$")


def detect_default_project_root():
    return Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split the global filtered-pairs CSV into one clean UTF-8 source-target CSV "
            "per language under data/<lang>/aligned/."
        )
    )
    parser.add_argument(
        "--run-tag",
        default=DEFAULT_RUN_TAG,
        help=f"Run tag used in reports/filtered_pairs_stats/<RUN_TAG>/ (default: {DEFAULT_RUN_TAG}).",
    )
    parser.add_argument(
        "--project-root",
        default=str(detect_default_project_root()),
        help="Project root (default: autodetected from src/).",
    )
    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        help=f"Reports directory relative to project root (default: {DEFAULT_REPORTS_DIR}).",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Data directory relative to project root (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--out-subdir",
        default=DEFAULT_OUT_SUBDIR,
        help=f"Per-language output subdirectory under data/<lang>/ (default: {DEFAULT_OUT_SUBDIR}).",
    )
    parser.add_argument(
        "--languages",
        default=DEFAULT_LANGS,
        help=f"Comma-separated languages to export (default: {DEFAULT_LANGS}).",
    )
    return parser.parse_args()


def parse_csv_list(raw_values):
    values = []
    seen = set()
    for token in raw_values.split(","):
        value = token.strip()
        if not value:
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("Could not parse a non-empty language list.")
    return values


def count_words(text):
    return len(str(text).split())


def extract_doc_number(doc_name):
    match = DOC_NUMBER_RE.search(doc_name)
    if match is None:
        raise ValueError(f"Could not derive num_doc from doc_name='{doc_name}'.")
    return int(match.group(1))


def resolve_input_csv(project_root, reports_dir, run_tag):
    return (
        Path(project_root)
        / reports_dir
        / run_tag
        / f"pairs_metrics_filtered_{run_tag}.csv"
    )


def load_rows_by_language(input_csv_path, languages):
    rows_by_language = {language: [] for language in languages}
    diagnostics = {
        language: {
            "rows_seen": 0,
            "rows_kept": 0,
            "rows_skipped_short_text": 0,
        }
        for language in languages
    }

    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            language = row["language"].strip()
            if language not in rows_by_language:
                continue

            diagnostics[language]["rows_seen"] += 1
            wiki_text = row["src_text"].strip()
            viki_text = row["tgt_text"].strip()

            if (
                count_words(wiki_text) <= MIN_WORDS_EXCLUSIVE
                or count_words(viki_text) <= MIN_WORDS_EXCLUSIVE
            ):
                diagnostics[language]["rows_skipped_short_text"] += 1
                continue

            rows_by_language[language].append(
                {
                    "num_doc": extract_doc_number(row["doc_name"].strip()),
                    "wiki_text": wiki_text,
                    "viki_text": viki_text,
                }
            )
            diagnostics[language]["rows_kept"] += 1

    for language in languages:
        rows_by_language[language].sort(
            key=lambda row: (row["num_doc"], row["wiki_text"], row["viki_text"])
        )

    return rows_by_language, diagnostics


def write_language_csv(path, rows):
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["num_doc", "wiki_text", "viki_text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    languages = parse_csv_list(args.languages)

    for language in languages:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Expected one of: {', '.join(SUPPORTED_LANGUAGES)}."
            )

    input_csv_path = resolve_input_csv(
        project_root=project_root,
        reports_dir=args.reports_dir,
        run_tag=args.run_tag,
    )
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    rows_by_language, diagnostics = load_rows_by_language(
        input_csv_path=input_csv_path,
        languages=languages,
    )

    for language in languages:
        out_dir = project_root / args.data_dir / language / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"final_filtered_pairs_{language}.csv"
        write_language_csv(out_path, rows_by_language[language])
        diag = diagnostics[language]
        print(
            f"[{language}] rows_seen={diag['rows_seen']} "
            f"rows_kept={diag['rows_kept']} "
            f"short_text_filtered={diag['rows_skipped_short_text']}"
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
