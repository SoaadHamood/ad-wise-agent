# build_one_row_per_category_csv.py
# Output CSV has EXACTLY 2 columns:
#   category, ads
# Where "ads" is a single long comma-separated string of all ads for that category.
# One output row per input file (category).
# Streaming write (does NOT hold all ads in memory).

import csv
import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Any


# =========================
# CONFIG (EDIT THESE)
# =========================

INPUT_ROOT = r"C:\data\amazon\scrapped_data\scrapped_data"          # folder that contains all category CSVs
OUTPUT_CSV = r"C:\data\amazon_category_ads.csv" #  2-column CSV

PREFERRED_AD_COLUMNS = [
    "ad", "Ad", "title", "Title", "headline", "Headline",
    "ad_title", "product_title", "text", "Text", "name", "Name"
]

ADS_SEPARATOR = ","  # between ads

REMOVE_INTERNAL_COMMAS = True

COLLAPSE_WHITESPACE = True
FILTER_JUNK = True

CSV_FIELD_LIMIT = 500 * 1024 * 1024  # 500MB


# =========================
# Helpers
# =========================

def set_csv_field_limit(limit: int) -> None:
    try:
        csv.field_size_limit(limit)
    except OverflowError:
        csv.field_size_limit(50 * 1024 * 1024)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(to_text(x) for x in value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def clean_text(value: Any) -> str:
    s = to_text(value)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    if COLLAPSE_WHITESPACE:
        s = re.sub(r"\s+", " ", s).strip()
    if REMOVE_INTERNAL_COMMAS:
        s = s.replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()
    return s


def is_junk_ad(ad: str) -> bool:
    t = (ad or "").strip().lower()
    if not t:
        return True
    junk_phrases = ["click to see price", "see price", "see details", "currently unavailable", "n/a"]
    if any(p in t for p in junk_phrases):
        return True
    if len(t) < 12:
        return True
    return False


def detect_ad_column(fieldnames: Optional[List[str]]) -> Optional[str]:
    if not fieldnames:
        return None
    lower_map = {c.lower(): c for c in fieldnames}
    for pref in PREFERRED_AD_COLUMNS:
        k = pref.lower()
        if k in lower_map:
            return lower_map[k]
    return None


def iter_ads_from_file(csv_path: Path) -> Iterator[str]:
    """
    Yield one ad per row from a category CSV file (streaming).
    """
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        ad_col = detect_ad_column(reader.fieldnames)

        for row in reader:
            if not row:
                continue

            if ad_col:
                ad = clean_text(row.get(ad_col, ""))
            else:
                # fallback: join all values
                parts = []
                for v in row.values():
                    txt = clean_text(v)
                    if txt:
                        parts.append(txt)
                ad = " | ".join(parts)

            if FILTER_JUNK and is_junk_ad(ad):
                continue

            if ad:
                yield ad


def csv_escape_field(s: str) -> str:
    """
    Escape a CSV field safely (always quote).
    Doubles internal quotes.
    """
    s = (s or "").replace('"', '""')
    return f'"{s}"'


def write_one_category_row_streaming(out_f, category: str, ads_iter: Iterator[str]) -> int:
    """
    Writes one CSV row:
      category, "ad1,ad2,ad3,..."
    Streaming: does not build the giant ads string in memory.
    Returns number of ads written.
    """
    ads_count = 0

    # write category field (quoted for safety)
    out_f.write(csv_escape_field(category))
    out_f.write(",")

    # start ads field (quoted)
    out_f.write('"')

    first = True
    for ad in ads_iter:
        # escape quotes inside ad
        ad = ad.replace('"', '""')

        if not first:
            out_f.write(ADS_SEPARATOR)
        out_f.write(ad)

        first = False
        ads_count += 1

    # end ads field + newline
    out_f.write('"\n')
    return ads_count


# =========================
# Main
# =========================

def main():
    set_csv_field_limit(CSV_FIELD_LIMIT)

    root = Path(INPUT_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {root}")

    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root}")

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        # header (exactly 2 columns)
        out_f.write("category,ads\n")

        for fp in files:
            category = fp.stem  # category = file name without .csv

            ads_iter = iter_ads_from_file(fp)
            count = write_one_category_row_streaming(out_f, category, ads_iter)

            print(f"✅ {fp.name} -> category='{category}' | ads_written={count:,}")

    print(f"\n✅ Done. Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()