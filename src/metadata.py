import csv
import io
import json
import os
import re
from typing import Dict, Optional

from openai import OpenAI


# -----------------------------
# Secrets helper
# -----------------------------
def _get_secret(name: str):
    """
    Works on Streamlit Cloud (st.secrets) and locally (.env / env vars).
    """
    try:
        import streamlit as st  # noqa: F401
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


# ============================================================
# CSV helpers (bulk override mode)
# ============================================================

def make_csv_template(filenames: list[str]) -> bytes:
    """
    Produces a CSV template for bulk metadata entry.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["filename", "source_url", "venue_name", "ensemble_name", "performance_date"]
    )
    for fn in filenames:
        writer.writerow([fn, "", "", "", ""])
    return buf.getvalue().encode("utf-8")


def parse_metadata_csv(csv_bytes: bytes) -> Dict[str, Dict]:
    """
    Parses uploaded CSV and returns:
      { filename: {source_url, venue_name, ensemble_name, performance_date} }
    """
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    required = {"filename", "source_url", "venue_name", "ensemble_name", "performance_date"}
    headers = set(reader.fieldnames or [])
    if not required.issubset(headers):
        raise ValueError(f"CSV must include headers: {sorted(required)}")

    out: Dict[str, Dict] = {}
    for row in reader:
        fn = (row.get("filename") or "").strip()
        if not fn:
            continue
        out[fn] = {
            "source_url": (row.get("source_url") or "").strip() or None,
            "venue_name": (row.get("venue_name") or "").strip() or None,
            "ensemble_name": (row.get("ensemble_name") or "").strip() or None,
            "performance_date": (row.get("performance_date") or "").strip() or None,
        }
    return out


def merge_metadata(
    filename: str,
    auto: Optional[Dict] = None,
    csv_data: Optional[Dict[str, Dict]] = None,
    overrides: Optional[Dict] = None,
) -> Dict:
    """
    Merge priority (highest wins):
      overrides > csv row > auto
    """
    auto = auto or {}
    overrides = overrides or {}
    row = (csv_data or {}).get(filename, {}) if csv_data else {}

    def pick(key: str):
        return (
            overrides.get(key)
            or row.get(key)
            or auto.get(key)
            or None
        )

    return {
        "source_url": pick("source_url"),
        "venue_name": pick("venue_name"),
        "ensemble_name": pick("ensemble_name"),
        "performance_date": pick("performance_date"),
    }


# ============================================================
# AI metadata auto-detect (all pages)
# ============================================================

URL_REGEX = re.compile(r"(https?://[^\s)>\]]+)", re.IGNORECASE)


_AUTODETECT_SYSTEM = (
    "You extract structured metadata from arts review / evidence PDFs for USCIS O-1 petitions. "
    "Return ONLY valid JSON. If a field is not found, return an empty string for that field."
)

_AUTODETECT_USER = """Extract metadata from the following document text.

Return JSON with keys:
- source_url
- venue_name
- ensemble_name
- performance_date

Guidelines:
- source_url: a URL visible in the document (prefer the publication URL).
- performance_date: the date of the performance/event (as written in the document).
- venue_name: venue / hall / festival / organisation hosting the performance.
- ensemble_name: orchestra/ensemble/choir/company performing (if stated).

DOCUMENT TEXT:
{text}
"""


def autodetect_metadata(
    document_text: str,
    *,
    model: Optional[str] = None,
    max_chars: int = 25000,
    debug: bool = False,
) -> Dict:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    chosen_model = model or _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    text = (document_text or "")
    prompt = _AUTODETECT_USER.format(text=text[:max_chars])

    data = {}
    try:
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": _AUTODETECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as e:
        print(f"[autodetect_metadata] Error: {e}")
        if debug:
            raise
        data = {}

    def s(key: str) -> str:
        val = data.get(key, "")
        return str(val or "").strip()

    # Lightweight URL fallback (helps even if model misses it)
    url = s("source_url")
    if not url:
        m = URL_REGEX.search(text)
        if m:
            url = m.group(1).strip().rstrip(".,);]")

    return {
        "source_url": url or "",
        "venue_name": s("venue_name"),
        "ensemble_name": s("ensemble_name"),
        "performance_date": s("performance_date"),
    }
