import argparse
import csv
import json
import re
import shutil
from pathlib import Path

DEFAULT_LANGS = "en,es,fr,it,ca"
DEFAULT_MODELS = "bge,labse,sonar"
DEFAULT_INPUT_STAGE = "01_raw_alignments"
DEFAULT_OUTPUT_STAGE = "02_filtered_alignments"
DEFAULT_SELECTION_MODE = "best_f1"
THRESHOLD_SOURCE = "experiments_analysis_strict_f1"

THRESHOLDS_REQUIRED_COLUMNS = {
    "language",
    "model",
    "min_score",
    "max_score",
    "f1_strict",
    "threshold_source",
    "source_note",
}
SUMMARY_FIELDS = [
    "run_tag",
    "language",
    "model",
    "min_score",
    "max_score",
    "f1_strict",
    "n_files_input",
    "n_files_output",
    "lines_total",
    "lines_kept",
    "lines_removed_low",
    "lines_removed_high",
    "invalid_lines",
    "kept_ratio",
]
PATH_LINE_RE = re.compile(
    r"^\[(?P<src>[^\]]*)\]:\[(?P<tgt>[^\]]*)\]:(?P<score>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def detect_default_project_root():
    return Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtra alineaciones full-corpus por thresholds strict versionados."
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Tag de corrida bajo runs/<RUN_TAG>/ (ej: 2026-02-28_fullCorpus_v1).",
    )
    parser.add_argument(
        "--project-root",
        default=str(detect_default_project_root()),
        help="Raiz del repo (default: autodetectado desde src/).",
    )
    parser.add_argument(
        "--thresholds-csv",
        default="config/best_thresholds_strict_fullcorpus.csv",
        help="CSV versionado con thresholds strict (default: config/best_thresholds_strict_fullcorpus.csv).",
    )
    parser.add_argument(
        "--languages",
        default=DEFAULT_LANGS,
        help="Idiomas separados por coma (default: en,es,fr,it,ca).",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Modelos separados por coma (default: bge,labse,sonar).",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["best_f1", "all_models"],
        default=DEFAULT_SELECTION_MODE,
        help=(
            "best_f1: un modelo por idioma segun mayor f1_strict en thresholds CSV "
            "(default). all_models: procesa todas las combinaciones idioma/modelo."
        ),
    )
    parser.add_argument(
        "--input-stage",
        default=DEFAULT_INPUT_STAGE,
        help="Stage de entrada (default: 01_raw_alignments).",
    )
    parser.add_argument(
        "--output-stage",
        default=DEFAULT_OUTPUT_STAGE,
        help="Stage de salida (default: 02_filtered_alignments).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribe outputs filtrados; solo genera resumen.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir stage de salida si ya existe.",
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
        raise ValueError("Se recibio una lista vacia tras parsear valores CSV.")
    return values


def load_thresholds(thresholds_csv_path):
    if not thresholds_csv_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de thresholds: {thresholds_csv_path}"
        )

    with thresholds_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = THRESHOLDS_REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"El CSV de thresholds no tiene columnas requeridas: {missing}"
            )

        table = {}
        rows = []
        for idx, row in enumerate(reader, start=2):
            language = row["language"].strip()
            model = row["model"].strip()
            key = (language, model)
            if not language or not model:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: language/model vacio."
                )
            if key in table:
                raise ValueError(
                    f"Fila duplicada para combination ({language}, {model}) en {thresholds_csv_path}."
                )

            try:
                min_score = float(row["min_score"])
                max_score = float(row["max_score"])
            except ValueError as exc:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: min_score/max_score no numerico."
                ) from exc

            if min_score > max_score:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: min_score > max_score."
                )

            source = row["threshold_source"].strip()
            if source != THRESHOLD_SOURCE:
                raise ValueError(
                    f"Fila {idx} invalida: threshold_source='{source}' (esperado '{THRESHOLD_SOURCE}')."
                )

            try:
                f1_strict = float(row["f1_strict"])
            except ValueError as exc:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: f1_strict no numerico."
                ) from exc

            n_docs_eval_raw = row.get("n_docs_eval", "").strip()
            n_docs_eval = None
            if n_docs_eval_raw:
                try:
                    n_docs_eval = int(float(n_docs_eval_raw))
                except ValueError as exc:
                    raise ValueError(
                        f"Fila {idx} invalida en {thresholds_csv_path}: n_docs_eval no numerico."
                    ) from exc

            record = {
                "language": language,
                "model": model,
                "min_score": min_score,
                "max_score": max_score,
                "f1_strict": f1_strict,
                "n_docs_eval": n_docs_eval,
                "threshold_source": source,
                "source_note": row["source_note"].strip(),
            }
            table[key] = record
            rows.append(record)

    return table, rows


def parse_score(line):
    match = PATH_LINE_RE.match(line.strip())
    if not match:
        return None
    try:
        return float(match.group("score"))
    except ValueError:
        return None


def filter_path_file(input_path, output_path, min_score, max_score, dry_run):
    lines_total = 0
    lines_kept = 0
    lines_removed_low = 0
    lines_removed_high = 0
    invalid_lines = 0
    kept_lines = []

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            lines_total += 1
            score = parse_score(raw_line)
            if score is None:
                invalid_lines += 1
                continue
            if score < min_score:
                lines_removed_low += 1
                continue
            if score > max_score:
                lines_removed_high += 1
                continue

            lines_kept += 1
            if not dry_run:
                kept_lines.append(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")

    if not dry_run:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.writelines(kept_lines)

    return {
        "lines_total": lines_total,
        "lines_kept": lines_kept,
        "lines_removed_low": lines_removed_low,
        "lines_removed_high": lines_removed_high,
        "invalid_lines": invalid_lines,
    }


def validate_inputs(
    project_root,
    run_tag,
    requested_pairs,
    input_stage,
    output_stage,
    thresholds_table,
    dry_run,
    overwrite,
):
    run_root = project_root / "runs" / run_tag
    if not run_root.exists():
        raise FileNotFoundError(f"No existe el run solicitado: {run_root}")

    errors = []
    combinations = []
    for language, model in requested_pairs:
        key = (language, model)
        if key not in thresholds_table:
            errors.append(
                f"Falta threshold para language={language}, model={model} en la tabla provista."
            )
            continue

        input_dir = run_root / language / input_stage / model
        output_dir = run_root / language / output_stage / model

        if not input_dir.exists():
            errors.append(f"No existe carpeta de input: {input_dir}")

        if output_dir.exists() and (not overwrite) and (not dry_run):
            errors.append(
                f"Output ya existe y no se paso --overwrite: {output_dir}"
            )

        combinations.append((language, model, input_dir, output_dir))

    if errors:
        raise RuntimeError("Validaciones fallaron:\n- " + "\n- ".join(errors))

    return run_root, combinations


def build_requested_pairs(languages, models, thresholds_table, selection_mode):
    if selection_mode == "all_models":
        return [(language, model) for language in languages for model in models]

    requested_pairs = []
    errors = []
    for language in languages:
        candidates = []
        for model in models:
            record = thresholds_table.get((language, model))
            if record is None:
                continue
            candidates.append(record)

        if not candidates:
            errors.append(
                f"No hay thresholds para language={language} en los modelos solicitados: {models}"
            )
            continue

        best = sorted(
            candidates,
            key=lambda row: (-row["f1_strict"], row["model"]),
        )[0]
        requested_pairs.append((language, best["model"]))
        print(
            f"[select best_f1] {language}: model={best['model']} "
            f"f1_strict={best['f1_strict']:.6f} "
            f"min={best['min_score']:.6f} max={best['max_score']:.6f}"
        )

    if errors:
        raise RuntimeError("Seleccion de modelos fallida:\n- " + "\n- ".join(errors))

    return requested_pairs


def write_summary_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(path, rows):
    payload = {"rows": rows}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_threshold_snapshot(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "language",
                "model",
                "min_score",
                "max_score",
                "f1_strict",
                "n_docs_eval",
                "threshold_source",
                "source_note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    thresholds_csv_path = Path(args.thresholds_csv)
    if not thresholds_csv_path.is_absolute():
        thresholds_csv_path = (project_root / thresholds_csv_path).resolve()

    languages = parse_csv_list(args.languages)
    models = parse_csv_list(args.models)
    thresholds_table, _ = load_thresholds(thresholds_csv_path)
    requested_pairs = build_requested_pairs(
        languages=languages,
        models=models,
        thresholds_table=thresholds_table,
        selection_mode=args.selection_mode,
    )

    run_root, combinations = validate_inputs(
        project_root=project_root,
        run_tag=args.run_tag,
        requested_pairs=requested_pairs,
        input_stage=args.input_stage,
        output_stage=args.output_stage,
        thresholds_table=thresholds_table,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    summary_rows = []
    for language, model, input_dir, output_dir in combinations:
        threshold = thresholds_table[(language, model)]
        min_score = threshold["min_score"]
        max_score = threshold["max_score"]

        input_files = sorted(input_dir.glob("*.path"))

        if not args.dry_run:
            if output_dir.exists() and args.overwrite:
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        counts = {
            "lines_total": 0,
            "lines_kept": 0,
            "lines_removed_low": 0,
            "lines_removed_high": 0,
            "invalid_lines": 0,
        }

        for input_file in input_files:
            output_file = output_dir / input_file.name
            file_counts = filter_path_file(
                input_path=input_file,
                output_path=output_file,
                min_score=min_score,
                max_score=max_score,
                dry_run=args.dry_run,
            )
            for key, value in file_counts.items():
                counts[key] += value

        n_files_output = len(input_files) if args.dry_run else len(list(output_dir.glob("*.path")))
        kept_ratio = (
            round(counts["lines_kept"] / counts["lines_total"], 6)
            if counts["lines_total"] > 0
            else 0.0
        )

        row = {
            "run_tag": args.run_tag,
            "language": language,
            "model": model,
            "min_score": f"{min_score:.6f}",
            "max_score": f"{max_score:.6f}",
            "f1_strict": f"{threshold['f1_strict']:.6f}",
            "n_files_input": len(input_files),
            "n_files_output": n_files_output,
            "lines_total": counts["lines_total"],
            "lines_kept": counts["lines_kept"],
            "lines_removed_low": counts["lines_removed_low"],
            "lines_removed_high": counts["lines_removed_high"],
            "invalid_lines": counts["invalid_lines"],
            "kept_ratio": f"{kept_ratio:.6f}",
        }
        summary_rows.append(row)

        print(
            f"[{language}/{model}] files={len(input_files)} "
            f"kept={counts['lines_kept']}/{counts['lines_total']} "
            f"invalid={counts['invalid_lines']}"
        )

    report_dir = project_root / "reports" / "full_corpus_filtering"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = report_dir / f"filter_summary_{args.run_tag}.csv"
    summary_json = report_dir / f"filter_summary_{args.run_tag}.json"
    write_summary_csv(summary_csv, summary_rows)
    write_summary_json(summary_json, summary_rows)

    if not args.dry_run:
        meta_dir = run_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = meta_dir / f"thresholds_applied_strict_{args.run_tag}.csv"
        effective_rows = [thresholds_table[(lang, model)] for (lang, model) in requested_pairs]
        write_threshold_snapshot(snapshot_path, effective_rows)
        print(f"Snapshot thresholds guardado en: {snapshot_path}")

    print(f"Resumen CSV: {summary_csv}")
    print(f"Resumen JSON: {summary_json}")


if __name__ == "__main__":
    main()
