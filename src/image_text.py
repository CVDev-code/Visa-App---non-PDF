from __future__ import annotations

import io
import os
import shutil
from typing import List, Dict, Any

from PIL import Image
import pytesseract


def _configure_tesseract() -> None:
    """
    pytesseract is a wrapper around the *system* `tesseract` binary.
    On Streamlit Cloud, install it via packages.txt.
    This function auto-discovers it (or uses TESSERACT_CMD env var).
    """
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return

    raise RuntimeError(
        "Tesseract OCR is not installed or not on PATH.\n\n"
        "Fix:\n"
        "1) On Streamlit Cloud: add a file `packages.txt` in repo root containing:\n"
        "   tesseract-ocr\n"
        "   tesseract-ocr-eng\n"
        "2) Or set env var TESSERACT_CMD to the full path of the tesseract binary."
    )


# Configure at import time so failures are immediate and obvious.
_configure_tesseract()


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    OCR the entire image and return plain text.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img) or ""


def get_ocr_words(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns OCR words with bounding boxes in pixel coords:
    [{"text": str, "x0": int, "y0": int, "x1": int, "y1": int, "conf": float}, ...]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    out: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        out.append(
            {
                "text": text,
                "x0": x,
                "y0": y,
                "x1": x + w,
                "y1": y + h,
                "conf": conf,
            }
        )
    return out
