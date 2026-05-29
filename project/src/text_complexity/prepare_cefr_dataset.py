from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import unicodedata
from urllib.request import urlopen

from .data import CEFR_TO_TARGET, normalize_cefr_level

DEFAULT_SOURCES = {
    "cambridge_exams_en": "https://huggingface.co/datasets/UniversalCEFR/cambridge_exams_en/resolve/main/cambridge_exams.json",
    "elg_cefr_en": "https://huggingface.co/datasets/UniversalCEFR/elg_cefr_en/resolve/main/elg-cefr-en.json",
}
DEFAULT_OUTPUT = "data/cefr_long_en.csv"


def _read_json_records(path: Path) -> list[dict[str, object]]:
    """Р§РёС‚Р°РµС‚ JSON РІ РЅРµСЃРєРѕР»СЊРєРёС… СЂР°СЃРїСЂРѕСЃС‚СЂР°РЅС‘РЅРЅС‹С… С„РѕСЂРјР°С‚Р°С… РёР· Hugging Face-РґР°С‚Р°СЃРµС‚РѕРІ."""

    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("train", "data", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"РќРµРїРѕРґРґРµСЂР¶РёРІР°РµРјР°СЏ СЃС‚СЂСѓРєС‚СѓСЂР° JSON РІ С„Р°Р№Р»Рµ {path}")


def download_raw_dataset(url: str, output_path: Path) -> None:
    """РЎРєР°С‡РёРІР°РµС‚ РёСЃС…РѕРґРЅС‹Р№ JSON СЂСЏРґРѕРј СЃ РїСЂРѕРµРєС‚РѕРј РґР»СЏ РІРѕСЃРїСЂРѕРёР·РІРѕРґРёРјРѕР№ РїРѕРґРіРѕС‚РѕРІРєРё РґР°РЅРЅС‹С…."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())


def _normalise_format(value: object) -> str:
    return str(value or "").strip().lower()


def _word_count(text: str) -> int:
    return len(text.split())


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """РЈРґР°Р»СЏРµС‚ РґСѓР±Р»РёРєР°С‚С‹ РїРѕСЃР»Рµ РЅРѕСЂРјР°Р»РёР·Р°С†РёРё РїСЂРѕР±РµР»РѕРІ, РєР°РІС‹С‡РµРє Рё СЂРµРіРёСЃС‚СЂР°."""

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = _normalize_text_for_dedupe(row["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _normalize_text_for_dedupe(text: str) -> str:
    """РџСЂРёРІРѕРґРёС‚ С‚РµРєСЃС‚ Рє РєР°РЅРѕРЅРёС‡РµСЃРєРѕРјСѓ РІРёРґСѓ РґР»СЏ СѓСЃС‚РѕР№С‡РёРІРѕРіРѕ dedupe."""

    normalized = unicodedata.normalize("NFKC", text)
    translation = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }
    )
    return " ".join(normalized.translate(translation).casefold().split())


def _select_rows(
    input_path: Path,
    max_per_level: int | None,
    formats: set[str],
    min_words: int,
) -> list[dict[str, str]]:
    """Р¤РёР»СЊС‚СЂСѓРµС‚ Р·Р°РїРёСЃРё РїРѕ С„РѕСЂРјР°С‚Сѓ, РґР»РёРЅРµ Рё Р»РёРјРёС‚Р°Рј РЅР° СѓСЂРѕРІРµРЅСЊ."""

    records = _read_json_records(input_path)
    selected_by_level: dict[str, int] = defaultdict(int)
    rows: list[dict[str, str]] = []
    for record in records:
        text = str(record.get("text", "")).strip()
        raw_level = str(record.get("cefr_level", "")).strip()
        if not text or not raw_level:
            continue
        text_format = _normalise_format(record.get("format", ""))
        if formats and text_format not in formats:
            continue
        if _word_count(text) < min_words:
            continue
        cefr_level = normalize_cefr_level(raw_level)
        if max_per_level is not None and selected_by_level[cefr_level] >= max_per_level:
            continue
        selected_by_level[cefr_level] += 1
        rows.append(
            {
                "text": text,
                "cefr_level": cefr_level,
                "target": str(CEFR_TO_TARGET[cefr_level]),
                "source_name": str(record.get("source_name", "cefr-sp")),
                "format": str(record.get("format", "")),
                "category": str(record.get("category", "")),
                "license": str(record.get("license", "CC BY-NC-SA 4.0")),
            }
        )
    return rows


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> int:
    """Р—Р°РїРёСЃС‹РІР°РµС‚ РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅС‹Рµ СЃС‚СЂРѕРєРё РІ РёС‚РѕРіРѕРІС‹Р№ CSV."""

    if not rows:
        raise ValueError("РќРµ РЅР°Р№РґРµРЅРѕ РїРѕРґС…РѕРґСЏС‰РёС… CEFR-СЃС‚СЂРѕРє РґР»СЏ Р·Р°РїРёСЃРё")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "text",
                "cefr_level",
                "target",
                "source_name",
                "format",
                "category",
                "license",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def convert_cefr_json_to_csv(
    input_path: Path,
    output_path: Path,
    max_per_level: int | None = None,
    formats: set[str] | None = None,
    min_words: int = 1,
) -> int:
    """РџСЂСЏРјР°СЏ РєРѕРЅРІРµСЂС‚Р°С†РёСЏ РѕРґРЅРѕРіРѕ JSON-С„Р°Р№Р»Р° РІ CSV."""

    rows = _select_rows(input_path, max_per_level, formats or set(), min_words)
    return write_rows(output_path, rows)


def prepare_long_dataset(
    raw_dir: Path,
    output_path: Path,
    skip_download: bool,
    long_max_per_level: int,
    min_words: int,
) -> int:
    """РЎРѕР±РёСЂР°РµС‚ РѕСЃРЅРѕРІРЅРѕР№ long-text РєРѕСЂРїСѓСЃ РґР»СЏ production-РјРѕРґРµР»Рё."""

    rows: list[dict[str, str]] = []
    for source_name in ("cambridge_exams_en", "elg_cefr_en"):
        raw_path = raw_dir / f"{source_name}.json"
        if not skip_download:
            download_raw_dataset(DEFAULT_SOURCES[source_name], raw_path)
        rows.extend(
            _select_rows(
                raw_path,
                max_per_level=long_max_per_level,
                formats={"document-level", "paragraph-level"},
                min_words=min_words,
            )
        )
    return write_rows(output_path, _dedupe_rows(rows))


def main() -> None:
    """CLI РґР»СЏ РїРѕРґРіРѕС‚РѕРІРєРё long РІРµСЂСЃРёРё CEFR-РєРѕСЂРїСѓСЃР°."""

    parser = argparse.ArgumentParser(description="РџРѕРґРіРѕС‚РѕРІРєР° CEFR-РґР°С‚Р°СЃРµС‚РѕРІ UniversalCEFR.")
    parser.add_argument("--raw-dir", default="data/raw", help="РљР°С‚Р°Р»РѕРі РґР»СЏ РёСЃС…РѕРґРЅС‹С… JSON-С„Р°Р№Р»РѕРІ.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="РџСѓС‚СЊ Рє РёС‚РѕРіРѕРІРѕРјСѓ CSV.")
    parser.add_argument("--long-max-per-level", type=int, default=1200, help="Р›РёРјРёС‚ long-text СЃС‚СЂРѕРє РЅР° СѓСЂРѕРІРµРЅСЊ.")
    parser.add_argument("--min-words", type=int, default=80, help="РњРёРЅРёРјР°Р»СЊРЅРѕРµ С‡РёСЃР»Рѕ СЃР»РѕРІ РґР»СЏ long-text СЃС‚СЂРѕРє.")
    parser.add_argument("--skip-download", action="store_true", help="РСЃРїРѕР»СЊР·РѕРІР°С‚СЊ СѓР¶Рµ СЃРєР°С‡Р°РЅРЅС‹Рµ JSON-С„Р°Р№Р»С‹.")
    args = parser.parse_args()

    count = prepare_long_dataset(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        skip_download=args.skip_download,
        long_max_per_level=args.long_max_per_level,
        min_words=args.min_words,
    )
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
