import argparse
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm

import spacy
import torch
from bert_score import score as bert_score

DEFAULT_LANGS = "en,es,fr,it,ca"
DEFAULT_MODELS = "bge,labse,sonar"
DEFAULT_SELECTION_MODE = "best_f1"
DEFAULT_THRESHOLDS_CSV = "config/best_thresholds_strict_fullcorpus.csv"
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
SPACY_MODEL_CANDIDATES = {
    "en": ["en_core_web_sm"],
    "es": ["es_core_news_sm"],
    "fr": ["fr_core_news_sm"],
    "it": ["it_core_news_sm"],
    "ca": ["ca_core_news_sm", "es_core_news_sm"],
}
ALIGN_RE = re.compile(
    r"^\[(?P<src>[^\]]*)\]:\[(?P<tgt>[^\]]*)\]:(?P<score>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
METRICS_MAIN = [
    "bertscore",
    "src_tree_depth",
    "tgt_tree_depth",
    "depth_reduction",
    "src_np_count",
    "tgt_np_count",
    "np_density_reduction",
    "src_word_count",
    "tgt_word_count",
    "word_count_reduction",
    "src_num_sents",
    "tgt_num_sents",
    "sentence_count_reduction",
]
METRICS_TOTAL = [
    "src_word_count",
    "tgt_word_count",
    "src_num_sents",
    "tgt_num_sents",
]


def detect_default_project_root():
    return Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analiza pares alineados filtrados en runs/<RUN_TAG>/<lang>/02_filtered_alignments/<model> "
            "y calcula BERTScore + metricas sintacticas."
        )
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Run tag a analizar (ej: 2026-02-28_fullCorpus_v1).",
    )
    parser.add_argument(
        "--project-root",
        default=str(detect_default_project_root()),
        help="Raiz del repo (default: autodetectado desde src/).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directorio de datos relativo a project-root (default: data).",
    )
    parser.add_argument(
        "--input-stage",
        default="02_filtered_alignments",
        help="Stage de entrada dentro del run (default: 02_filtered_alignments).",
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
            "(default). all_models: analiza todas las combinaciones idioma/modelo."
        ),
    )
    parser.add_argument(
        "--thresholds-csv",
        default=DEFAULT_THRESHOLDS_CSV,
        help="CSV de thresholds con f1_strict usado por selection-mode=best_f1.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="Score maximo permitido (inclusivo) para considerar pares (default: 0.95).",
    )
    parser.add_argument(
        "--bertscore-model",
        default="xlm-roberta-large",
        help="Modelo para BERTScore (default: xlm-roberta-large).",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=32,
        help="Batch size de BERTScore (default: 32).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo para BERTScore: auto|cpu|cuda (default: auto).",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=0,
        help="Limite opcional por language+model (0 = sin limite).",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Salta calculo de BERTScore (debug rapido).",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/filtered_pairs_stats",
        help="Carpeta base de salida (default: reports/filtered_pairs_stats).",
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
        raise ValueError("No se pudo parsear una lista valida desde el valor recibido.")
    return values


def load_thresholds_table(thresholds_csv_path):
    if not thresholds_csv_path.exists():
        raise FileNotFoundError(f"No existe thresholds CSV: {thresholds_csv_path}")

    table = {}
    with thresholds_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = THRESHOLDS_REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"Thresholds CSV sin columnas requeridas: {', '.join(sorted(missing))}"
            )

        for idx, row in enumerate(reader, start=2):
            language = row["language"].strip()
            model = row["model"].strip()
            if not language or not model:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: language/model vacio."
                )
            key = (language, model)
            if key in table:
                raise ValueError(
                    f"Fila duplicada para ({language}, {model}) en {thresholds_csv_path}"
                )

            source = row["threshold_source"].strip()
            if source != THRESHOLD_SOURCE:
                raise ValueError(
                    f"Fila {idx} invalida: threshold_source='{source}' (esperado '{THRESHOLD_SOURCE}')"
                )

            try:
                f1_strict = float(row["f1_strict"])
            except ValueError as exc:
                raise ValueError(
                    f"Fila {idx} invalida en {thresholds_csv_path}: f1_strict no numerico."
                ) from exc

            table[key] = {
                "language": language,
                "model": model,
                "f1_strict": f1_strict,
            }

    return table


def build_requested_pairs(languages, models, selection_mode, thresholds_table):
    if selection_mode == "all_models":
        return [(language, model) for language in languages for model in models]

    requested_pairs = []
    errors = []
    for language in languages:
        candidates = []
        for model in models:
            row = thresholds_table.get((language, model))
            if row is not None:
                candidates.append(row)

        if not candidates:
            errors.append(
                f"Sin filas de thresholds para language={language} en modelos={models}"
            )
            continue

        best = sorted(candidates, key=lambda x: (-x["f1_strict"], x["model"]))[0]
        requested_pairs.append((language, best["model"]))
        print(
            f"[select best_f1] {language}: model={best['model']} "
            f"f1_strict={best['f1_strict']:.6f}"
        )

    if errors:
        raise RuntimeError("Seleccion de modelos fallida:\n- " + "\n- ".join(errors))

    return requested_pairs


def parse_index_block(text):
    text = text.strip()
    if not text:
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def parse_alignment_line(line):
    match = ALIGN_RE.match(line.strip())
    if match is None:
        return None

    src_idx = parse_index_block(match.group("src"))
    tgt_idx = parse_index_block(match.group("tgt"))
    score = float(match.group("score"))
    return src_idx, tgt_idx, score


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def build_side_text(lines, indices):
    sents = [lines[i].strip() for i in indices if 0 <= i < len(lines) and lines[i].strip()]
    return sents, " ".join(sents)


def make_diag_row(language, model):
    return {
        "language": language,
        "model": model,
        "n_files_input": 0,
        "n_lines_total": 0,
        "n_lines_invalid": 0,
        "n_lines_skipped_over_max": 0,
        "n_lines_skipped_empty_side": 0,
        "n_lines_kept": 0,
        "n_missing_docs": 0,
        "n_rows_emitted": 0,
    }


def collect_pairs_for_doc(language, model, path_file, data_dir, max_score):
    doc_name = path_file.name.replace(".path", "")
    src_file = data_dir / language / "wiki" / doc_name
    tgt_file = data_dir / language / "viki" / doc_name

    if not src_file.exists() or not tgt_file.exists():
        return [], True, 0, 0, 0, 0

    src_lines = read_lines(src_file)
    tgt_lines = read_lines(tgt_file)

    rows = []
    invalid = 0
    over_max = 0
    empty_side = 0
    total = 0

    for line_no, raw in enumerate(path_file.read_text(encoding="utf-8").splitlines(), start=1):
        total += 1
        parsed = parse_alignment_line(raw)
        if parsed is None:
            invalid += 1
            continue

        src_idx, tgt_idx, score = parsed
        if score > max_score:
            over_max += 1
            continue

        src_sents, src_text = build_side_text(src_lines, src_idx)
        tgt_sents, tgt_text = build_side_text(tgt_lines, tgt_idx)

        if not src_sents or not tgt_sents:
            empty_side += 1
            continue

        rows.append(
            {
                "language": language,
                "model": model,
                "doc_name": doc_name,
                "line_in_path": line_no,
                "score": score,
                "src_indices": src_idx,
                "tgt_indices": tgt_idx,
                "src_sents": src_sents,
                "tgt_sents": tgt_sents,
                "src_text": src_text,
                "tgt_text": tgt_text,
                "src_num_sents": len(src_sents),
                "tgt_num_sents": len(tgt_sents),
            }
        )

    return rows, False, total, invalid, over_max, empty_side


def collect_all_pairs(project_root, run_tag, input_stage, data_dir, requested_pairs, max_score):
    run_root = project_root / "runs" / run_tag
    if not run_root.exists():
        raise FileNotFoundError(f"No existe run tag: {run_root}")

    rows = []
    diag_rows = []

    for language, model in requested_pairs:
        diag = make_diag_row(language, model)
        input_dir = run_root / language / input_stage / model
        if not input_dir.exists():
            raise FileNotFoundError(f"No existe carpeta de input: {input_dir}")

        path_files = sorted(input_dir.glob("*.path"))
        diag["n_files_input"] = len(path_files)

        for path_file in path_files:
            doc_rows, missing_doc, total, invalid, over_max, empty_side = collect_pairs_for_doc(
                language=language,
                model=model,
                path_file=path_file,
                data_dir=data_dir,
                max_score=max_score,
            )
            if missing_doc:
                diag["n_missing_docs"] += 1
                continue

            rows.extend(doc_rows)
            diag["n_lines_total"] += total
            diag["n_lines_invalid"] += invalid
            diag["n_lines_skipped_over_max"] += over_max
            diag["n_lines_skipped_empty_side"] += empty_side
            diag["n_lines_kept"] += len(doc_rows)

        diag["n_rows_emitted"] = diag["n_lines_kept"]
        diag_rows.append(diag)

        print(
            f"[collect {language}/{model}] files={diag['n_files_input']} "
            f"kept={diag['n_rows_emitted']} invalid={diag['n_lines_invalid']} "
            f"missing_docs={diag['n_missing_docs']}"
        )

    pairs_df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows).sort_values(["language", "model"]).reset_index(drop=True)
    return pairs_df, diag_df


def load_spacy_pipelines(languages):
    nlp_by_lang = {}
    missing = {}

    for language in languages:
        loaded = None
        tried = []

        for model_name in SPACY_MODEL_CANDIDATES.get(language, []):
            tried.append(model_name)
            try:
                loaded = spacy.load(model_name, disable=["lemmatizer", "textcat", "ner"])
                print(f"[{language}] usando spaCy model: {model_name}")
                break
            except Exception:
                continue

        if loaded is None:
            missing[language] = tried
        else:
            nlp_by_lang[language] = loaded

    return nlp_by_lang, missing


def doc_tree_depth(doc):
    sent_depths = []
    for sent in doc.sents:
        depths = [len(list(tok.ancestors)) + 1 for tok in sent if not tok.is_space]
        if depths:
            sent_depths.append(max(depths))
    if not sent_depths:
        return np.nan
    return float(np.mean(sent_depths))


def doc_np_count(doc):
    try:
        return float(sum(1 for _ in doc.noun_chunks))
    except Exception:
        return float(sum(1 for tok in doc if tok.pos_ in {"NOUN", "PROPN"}))


def doc_token_count(doc):
    return float(sum(1 for tok in doc if not tok.is_space))


def text_syntax_metrics(language, text, nlp_by_lang, syntax_cache):
    key = (language, text)
    cached = syntax_cache.get(key)
    if cached is not None:
        return cached

    doc = nlp_by_lang[language](text)
    tree_depth = doc_tree_depth(doc)
    np_count = doc_np_count(doc)
    token_count = doc_token_count(doc)
    out = (tree_depth, np_count, token_count)
    syntax_cache[key] = out
    return out


def side_metrics(language, sents, nlp_by_lang, syntax_cache):
    if not sents:
        return np.nan, np.nan, np.nan, np.nan

    depths = []
    np_counts = []
    token_counts = []

    for sent in sents:
        depth, np_count, token_count = text_syntax_metrics(
            language=language,
            text=sent,
            nlp_by_lang=nlp_by_lang,
            syntax_cache=syntax_cache,
        )
        if not np.isnan(depth):
            depths.append(depth)
        np_counts.append(np_count)
        token_counts.append(token_count)

    side_depth = float(np.mean(depths)) if depths else np.nan
    side_np_count = float(np.sum(np_counts))
    side_token_count = float(np.sum(token_counts))
    side_np_density = side_np_count / side_token_count if side_token_count > 0 else np.nan
    return side_depth, side_np_count, side_token_count, side_np_density


def add_syntax_metrics(df, nlp_by_lang):
    out = df.copy()
    syntax_cache = {}

    src_depths = []
    tgt_depths = []
    src_np_counts = []
    tgt_np_counts = []
    src_token_counts = []
    tgt_token_counts = []
    src_np_densities = []
    tgt_np_densities = []

    for row in tqdm(out.itertuples(index=False), total=len(out), desc="spaCy metrics"):
        src_d, src_n, src_tok, src_den = side_metrics(
            language=row.language,
            sents=row.src_sents,
            nlp_by_lang=nlp_by_lang,
            syntax_cache=syntax_cache,
        )
        tgt_d, tgt_n, tgt_tok, tgt_den = side_metrics(
            language=row.language,
            sents=row.tgt_sents,
            nlp_by_lang=nlp_by_lang,
            syntax_cache=syntax_cache,
        )

        src_depths.append(src_d)
        tgt_depths.append(tgt_d)
        src_np_counts.append(src_n)
        tgt_np_counts.append(tgt_n)
        src_token_counts.append(src_tok)
        tgt_token_counts.append(tgt_tok)
        src_np_densities.append(src_den)
        tgt_np_densities.append(tgt_den)

    out["src_tree_depth"] = src_depths
    out["tgt_tree_depth"] = tgt_depths
    out["depth_reduction"] = out["src_tree_depth"] - out["tgt_tree_depth"]
    out["src_np_count"] = src_np_counts
    out["tgt_np_count"] = tgt_np_counts
    out["src_token_count"] = src_token_counts
    out["tgt_token_count"] = tgt_token_counts
    out["src_np_density"] = src_np_densities
    out["tgt_np_density"] = tgt_np_densities
    out["np_density_reduction"] = out["src_np_density"] - out["tgt_np_density"]
    # Re-export token counts as word counts for reporting consistency.
    out["src_word_count"] = out["src_token_count"]
    out["tgt_word_count"] = out["tgt_token_count"]
    out["word_count_reduction"] = out["src_word_count"] - out["tgt_word_count"]
    out["src_num_sents"] = pd.to_numeric(out["src_num_sents"], errors="coerce")
    out["tgt_num_sents"] = pd.to_numeric(out["tgt_num_sents"], errors="coerce")
    out["sentence_count_reduction"] = out["src_num_sents"] - out["tgt_num_sents"]
    return out


def add_bertscore(df, bertscore_model, bertscore_batch_size, device):
    out = df.copy()
    out["bertscore"] = np.nan

    for (language, model), idx in out.groupby(["language", "model"]).groups.items():
        idx = list(idx)
        refs = out.loc[idx, "src_text"].tolist()
        cands = out.loc[idx, "tgt_text"].tolist()
        if not refs:
            continue

        print(f"BERTScore -> lang={language}, model={model}, n={len(idx)}")
        _, _, f1 = bert_score(
            cands=cands,
            refs=refs,
            model_type=bertscore_model,
            batch_size=bertscore_batch_size,
            device=device,
            verbose=False,
        )
        out.loc[idx, "bertscore"] = f1.detach().cpu().numpy()

    return out


def aggregate_stats(df):
    grouped = df.groupby(["language", "model"])

    stats_df = grouped[METRICS_MAIN].agg(["mean", "median", "min", "max"]).reset_index()
    stats_df.columns = [
        "language",
        "model",
        *[f"{metric}_{stat}" for metric in METRICS_MAIN for stat in ["mean", "median", "min", "max"]],
    ]
    totals_df = grouped[METRICS_TOTAL].sum(min_count=1).reset_index()
    totals_df = totals_df.rename(columns={metric: f"{metric}_total" for metric in METRICS_TOTAL})
    stats_df = stats_df.merge(totals_df, on=["language", "model"], how="left")
    stats_df = stats_df.sort_values(["language", "model"]).reset_index(drop=True)
    return stats_df


def fit_normals(df):
    rows = []
    for (language, model), group in df.groupby(["language", "model"]):
        row = {"language": language, "model": model}
        for metric in METRICS_MAIN:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().values
            if len(vals) == 0:
                mu, sigma = np.nan, np.nan
            elif len(vals) == 1:
                mu, sigma = float(vals[0]), 0.0
            else:
                mu, sigma = norm.fit(vals)
            row[f"{metric}_mu"] = mu
            row[f"{metric}_sigma"] = sigma
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["language", "model"]).reset_index(drop=True)


def make_score_check(df):
    return (
        df.groupby(["language", "model"])["score"]
        .agg(n_pairs="count", min_score="min", max_score="max")
        .reset_index()
        .sort_values(["language", "model"])
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()

    languages = parse_csv_list(args.languages)
    models = parse_csv_list(args.models)
    thresholds_table = {}
    thresholds_csv_path = Path(args.thresholds_csv)
    if not thresholds_csv_path.is_absolute():
        thresholds_csv_path = (project_root / thresholds_csv_path).resolve()
    if args.selection_mode == "best_f1":
        thresholds_table = load_thresholds_table(thresholds_csv_path)
    requested_pairs = build_requested_pairs(
        languages=languages,
        models=models,
        selection_mode=args.selection_mode,
        thresholds_table=thresholds_table,
    )

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    print("PROJECT_ROOT:", project_root)
    print("DATA_DIR:", data_dir)
    print("RUN_TAG:", args.run_tag)
    print("INPUT_STAGE:", args.input_stage)
    print("LANGUAGES:", languages)
    print("MODELS (candidatos):", models)
    print("SELECTION_MODE:", args.selection_mode)
    print("REQUESTED_PAIRS:", requested_pairs)
    print("BERTScore model:", args.bertscore_model, "| device:", device)

    pairs_df, diag_df = collect_all_pairs(
        project_root=project_root,
        run_tag=args.run_tag,
        input_stage=args.input_stage,
        data_dir=data_dir,
        requested_pairs=requested_pairs,
        max_score=args.max_score,
    )
    if pairs_df.empty:
        raise RuntimeError("No se recuperaron pares para analizar.")

    if args.max_pairs_per_group > 0:
        pairs_df = (
            pairs_df.sort_values(["language", "model", "doc_name", "line_in_path"])
            .groupby(["language", "model"], as_index=False, group_keys=False)
            .head(args.max_pairs_per_group)
            .reset_index(drop=True)
        )
        print("Aplicado max_pairs_per_group:", args.max_pairs_per_group)
        print("Pares tras recorte:", len(pairs_df))

    nlp_by_lang, missing_langs = load_spacy_pipelines(sorted(pairs_df["language"].unique()))
    if missing_langs:
        for language, tried in sorted(missing_langs.items()):
            warnings.warn(f"Sin modelo spaCy para {language}. Intentados: {tried}")

    usable_langs = sorted(nlp_by_lang.keys())
    pairs_df = pairs_df[pairs_df["language"].isin(usable_langs)].reset_index(drop=True)
    if pairs_df.empty:
        raise RuntimeError("Sin pares utilizables tras filtrar idiomas sin spaCy.")

    pairs_metrics_df = add_syntax_metrics(pairs_df, nlp_by_lang)
    if args.skip_bertscore:
        pairs_metrics_df["bertscore"] = np.nan
        print("BERTScore omitido por --skip-bertscore")
    else:
        pairs_metrics_df = add_bertscore(
            pairs_metrics_df,
            bertscore_model=args.bertscore_model,
            bertscore_batch_size=args.bertscore_batch_size,
            device=device,
        )

    stats_df = aggregate_stats(pairs_metrics_df)
    norm_df = fit_normals(pairs_metrics_df)
    score_check_df = make_score_check(pairs_metrics_df)

    out_dir = (project_root / args.out_dir / args.run_tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_export_df = pairs_metrics_df.copy()
    pairs_export_df["src_indices"] = pairs_export_df["src_indices"].apply(lambda x: ",".join(map(str, x)))
    pairs_export_df["tgt_indices"] = pairs_export_df["tgt_indices"].apply(lambda x: ",".join(map(str, x)))
    pairs_export_df["src_sents"] = pairs_export_df["src_sents"].apply(lambda xs: " || ".join(xs))
    pairs_export_df["tgt_sents"] = pairs_export_df["tgt_sents"].apply(lambda xs: " || ".join(xs))

    pairs_path = out_dir / f"pairs_metrics_filtered_{args.run_tag}.csv"
    stats_path = out_dir / f"summary_stats_filtered_{args.run_tag}.csv"
    norm_path = out_dir / f"normal_fit_filtered_{args.run_tag}.csv"
    diag_path = out_dir / f"input_diagnostics_filtered_{args.run_tag}.csv"
    score_check_path = out_dir / f"score_check_filtered_{args.run_tag}.csv"

    pairs_export_df.to_csv(pairs_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    norm_df.to_csv(norm_path, index=False)
    diag_df.to_csv(diag_path, index=False)
    score_check_df.to_csv(score_check_path, index=False)

    print("Guardado:")
    print("-", pairs_path)
    print("-", stats_path)
    print("-", norm_path)
    print("-", diag_path)
    print("-", score_check_path)


if __name__ == "__main__":
    main()
