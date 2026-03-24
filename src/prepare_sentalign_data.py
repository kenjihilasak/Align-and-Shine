import argparse
import csv
import json
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from time import perf_counter

DEFAULT_LANGS = "es,en,fr,it,ca"
MAX_OPEN_FILES = 256
SUMMARY_FIELDS = [
    "lang",
    "wiki_docs",
    "viki_docs",
    "paired_docs",
    "generated_docs",
    "skipped_docs",
    "wiki_sent_rows",
    "viki_sent_rows",
    "duration_s",
    "gpu_detected",
]


class LRUFileWriter:
    def __init__(self, max_open_files=MAX_OPEN_FILES):
        self.max_open_files = max_open_files
        self._handles = OrderedDict()

    def write_line(self, path, line):
        key = str(path)
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
        self._handles[key] = handle
        if len(self._handles) > self.max_open_files:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        handle.write(line)
        handle.write("\n")

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera TXT de pares Wiki/Vikidia para todos los documentos disponibles."
    )
    parser.add_argument(
        "--langs",
        default=DEFAULT_LANGS,
        help="Idiomas separados por coma (default: es,en,fr,it,ca)",
    )
    parser.add_argument(
        "--all-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Procesar todos los documentos pareados (default: true)",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Borrar *.txt existentes en data/<lang>/wiki|viki antes de regenerar (default: true)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Raiz de datos (default: data)",
    )
    parser.add_argument(
        "--max-docs-per-lang",
        type=int,
        default=0,
        help="Limite de docs por idioma para smoke tests (0 = sin limite)",
    )
    parser.add_argument(
        "--gpu-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detectar y registrar disponibilidad de GPU (default: true)",
    )
    parser.add_argument(
        "--doc-ids",
        default="",
        help="Compatibilidad legacy: lista de IDs separada por coma (ej: 1260,1725)",
    )
    return parser.parse_args()


def normalize_doc_id(raw_doc_id):
    text = str(raw_doc_id).strip()
    for prefix in ("wiki-", "viki-", "doc-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith(".txt"):
        text = text[:-4]
    return text.strip()


def parse_langs(raw_langs):
    langs = [token.strip() for token in raw_langs.split(",") if token.strip()]
    if not langs:
        raise ValueError("No se recibieron idiomas validos en --langs.")
    # Mantener orden, deduplicando
    seen = set()
    ordered = []
    for lang in langs:
        if lang not in seen:
            ordered.append(lang)
            seen.add(lang)
    return ordered


def doc_sort_key(doc_id):
    if doc_id.isdigit():
        return (0, int(doc_id))
    return (1, doc_id)


def parse_doc_ids(raw_doc_ids):
    if not raw_doc_ids.strip():
        return []
    ids = []
    seen = set()
    for token in raw_doc_ids.split(","):
        doc_id = normalize_doc_id(token)
        if doc_id and doc_id not in seen:
            ids.append(doc_id)
            seen.add(doc_id)
    return ids


def detect_gpu_status():
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return "true", f"torch cuda available ({count} GPU, first: {torch.cuda.get_device_name(0)})"
            return "true", "torch cuda available"
        return "false", "torch instalado pero cuda no disponible"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            return "true", f"nvidia-smi detecta GPU ({first_line})"
        if result.returncode == 0:
            return "false", "nvidia-smi sin dispositivos listados"
        return "false", f"nvidia-smi retorno codigo {result.returncode}"
    except FileNotFoundError:
        return "unknown", "torch/nvidia-smi no disponibles"


def scan_doc_ids(jsonl_path):
    doc_ids = set()
    rows = 0
    bad_lines = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            doc_id = normalize_doc_id(obj.get("doc_id", ""))
            if doc_id:
                doc_ids.add(doc_id)

    return doc_ids, rows, bad_lines


def clean_txt_files(directory):
    removed = 0
    if not directory.exists():
        return removed
    for txt_file in directory.glob("*.txt"):
        txt_file.unlink()
        removed += 1
    return removed


def clear_selected_files(directory, selected_doc_ids):
    removed = 0
    for doc_id in selected_doc_ids:
        out_path = directory / f"doc-{doc_id}.txt"
        if out_path.exists():
            out_path.unlink()
            removed += 1
    return removed


def write_side_streaming(jsonl_path, selected_doc_ids, output_dir):
    writer = LRUFileWriter()
    rows_seen = 0
    sentences_written = 0
    docs_written = set()
    bad_lines = 0

    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows_seen += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue

                doc_id = normalize_doc_id(obj.get("doc_id", ""))
                if doc_id not in selected_doc_ids:
                    continue

                sentence = str(obj.get("sentence", ""))
                out_path = output_dir / f"doc-{doc_id}.txt"
                writer.write_line(out_path, sentence)
                sentences_written += 1
                docs_written.add(doc_id)
    finally:
        writer.close()

    return rows_seen, sentences_written, docs_written, bad_lines


def build_selection(paired_doc_ids, args):
    requested_doc_ids = parse_doc_ids(args.doc_ids)

    if requested_doc_ids:
        selected = [doc_id for doc_id in requested_doc_ids if doc_id in paired_doc_ids]
        missing = [doc_id for doc_id in requested_doc_ids if doc_id not in paired_doc_ids]
        if missing:
            print(f"  [WARN] {len(missing)} doc_ids no pareados/ausentes y se omiten.")
    elif args.all_docs:
        selected = sorted(paired_doc_ids, key=doc_sort_key)
    else:
        selected = []

    if args.max_docs_per_lang > 0:
        selected = selected[: args.max_docs_per_lang]

    return selected


def process_language(lang, args, gpu_detected):
    start = perf_counter()
    data_root = Path(args.data_root)
    wiki_raw = data_root / lang / "raw" / f"filtered_wiki-{lang}.sentences.jsonl"
    viki_raw = data_root / lang / "raw" / f"filtered_vikidia-{lang}.sentences.jsonl"
    wiki_out_dir = data_root / lang / "wiki"
    viki_out_dir = data_root / lang / "viki"

    print(f"\n== [{lang}] ==")
    print(f"  Wiki raw: {wiki_raw}")
    print(f"  Viki raw: {viki_raw}")

    if not wiki_raw.exists() or not viki_raw.exists():
        print("  [ERROR] Archivos raw faltantes. Saltando idioma.")
        return {
            "lang": lang,
            "wiki_docs": 0,
            "viki_docs": 0,
            "paired_docs": 0,
            "generated_docs": 0,
            "skipped_docs": 0,
            "wiki_sent_rows": 0,
            "viki_sent_rows": 0,
            "duration_s": 0.0,
            "gpu_detected": gpu_detected,
        }

    wiki_doc_ids, wiki_rows, wiki_bad = scan_doc_ids(wiki_raw)
    viki_doc_ids, viki_rows, viki_bad = scan_doc_ids(viki_raw)
    paired_doc_ids = wiki_doc_ids.intersection(viki_doc_ids)
    selected_doc_ids = build_selection(paired_doc_ids, args)

    print(
        f"  Docs -> wiki={len(wiki_doc_ids)} viki={len(viki_doc_ids)} "
        f"pareados={len(paired_doc_ids)} seleccionados={len(selected_doc_ids)}"
    )
    if wiki_bad or viki_bad:
        print(f"  [WARN] JSONL con lineas invalidas (wiki={wiki_bad}, viki={viki_bad}).")

    wiki_out_dir.mkdir(parents=True, exist_ok=True)
    viki_out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        removed_wiki = clean_txt_files(wiki_out_dir)
        removed_viki = clean_txt_files(viki_out_dir)
        print(f"  Limpieza global: wiki={removed_wiki} txt, viki={removed_viki} txt.")
    else:
        removed_wiki = clear_selected_files(wiki_out_dir, selected_doc_ids)
        removed_viki = clear_selected_files(viki_out_dir, selected_doc_ids)
        print(f"  Limpieza parcial (docs seleccionados): wiki={removed_wiki}, viki={removed_viki}.")

    if not selected_doc_ids:
        print("  [SKIP] No hay documentos seleccionados para generar.")
        duration_s = round(perf_counter() - start, 3)
        return {
            "lang": lang,
            "wiki_docs": len(wiki_doc_ids),
            "viki_docs": len(viki_doc_ids),
            "paired_docs": 0,
            "generated_docs": 0,
            "skipped_docs": 0,
            "wiki_sent_rows": wiki_rows,
            "viki_sent_rows": viki_rows,
            "duration_s": duration_s,
            "gpu_detected": gpu_detected,
        }

    selected_set = set(selected_doc_ids)

    _, wiki_sent_written, wiki_docs_written, wiki_bad_write = write_side_streaming(
        wiki_raw, selected_set, wiki_out_dir
    )
    _, viki_sent_written, viki_docs_written, viki_bad_write = write_side_streaming(
        viki_raw, selected_set, viki_out_dir
    )

    generated_docs = len(wiki_docs_written.intersection(viki_docs_written))
    skipped_docs = len(selected_set) - generated_docs

    wiki_present = sum((wiki_out_dir / f"doc-{doc_id}.txt").exists() for doc_id in selected_set)
    viki_present = sum((viki_out_dir / f"doc-{doc_id}.txt").exists() for doc_id in selected_set)
    if wiki_present != viki_present:
        print(f"  [WARN] Desbalance de pares en disco (wiki={wiki_present}, viki={viki_present}).")

    if wiki_bad_write or viki_bad_write:
        print(
            f"  [WARN] Lineas invalidas durante escritura (wiki={wiki_bad_write}, "
            f"viki={viki_bad_write})."
        )

    duration_s = round(perf_counter() - start, 3)
    print(
        f"  [OK] docs_generados={generated_docs} skipped={skipped_docs} "
        f"wiki_sents={wiki_sent_written} viki_sents={viki_sent_written} tiempo={duration_s}s"
    )

    return {
        "lang": lang,
        "wiki_docs": len(wiki_doc_ids),
        "viki_docs": len(viki_doc_ids),
        "paired_docs": len(selected_set),
        "generated_docs": generated_docs,
        "skipped_docs": skipped_docs,
        "wiki_sent_rows": wiki_rows,
        "viki_sent_rows": viki_rows,
        "duration_s": duration_s,
        "gpu_detected": gpu_detected,
    }


def write_summary(rows):
    report_dir = Path("reports") / "prepare_data"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"prepare_summary_{timestamp}.csv"

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return report_path


def prepare():
    args = parse_args()
    langs = parse_langs(args.langs)

    print("=== PREPARE SENTALIGN DATA (ALL DOCS) ===")
    print(f"Idiomas: {langs}")
    print(
        f"Opciones: all_docs={args.all_docs} clean_output={args.clean_output} "
        f"max_docs_per_lang={args.max_docs_per_lang} data_root={args.data_root}"
    )

    if args.gpu_check:
        gpu_detected, gpu_note = detect_gpu_status()
        print(f"GPU check: {gpu_detected} ({gpu_note})")
    else:
        gpu_detected = "unknown"
        print("GPU check: disabled")

    rows = []
    for lang in langs:
        rows.append(process_language(lang, args, gpu_detected))

    report_path = write_summary(rows)
    total_pairs = sum(row["generated_docs"] for row in rows)
    print("\n=== RESUMEN FINAL ===")
    print(f"Docs pareados generados totales: {total_pairs}")
    print(f"Reporte CSV: {report_path}")
    print("¡Listo!")


if __name__ == "__main__":
    prepare()
