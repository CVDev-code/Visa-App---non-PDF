from __future__ import annotations

import io
from typing import Tuple

from PIL import Image
import pytesseract


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    OCR the entire image and return plain text.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img) or ""


def get_ocr_words(image_bytes: bytes):
    """
    Returns OCR words with bounding boxes in pixel coords:
    [{"text": str, "x0": int, "y0": int, "x1": int, "y1": int, "conf": float}, ...]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    out = []
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
