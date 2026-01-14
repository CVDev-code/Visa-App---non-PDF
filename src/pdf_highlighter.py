"""
Dynamic callout placement + obstacle-aware routing for PDF annotations (PyMuPDF).

What this rewrite changes (while keeping your core logic intact):
- Keeps: criteria-dependent callouts (including star criteria), URL-only matching restricted to page 1,
  multi-page gutter routing for non-page-1 targets, A* routing on page 1, arrows drawn first then callouts.
- Replaces: fixed “gutter-only” placement with **dynamic near-target placement** (tries multiple candidate
  positions around the red box / target union, scores candidates using your own A* router and overlap penalties).
- Keeps: gutters as a robust fallback if no good near-target candidate exists.
- Enforces: font size never below 10.

Notes:
- The algorithm does NOT rely on “white space” being empty in the PDF object model.
  It places callouts by avoiding text boxes and red boxes (hard obstacles) and penalizing overlap with text (soft).
- Optional hook included to later integrate render-to-image whitespace masks if you want.
"""

import io
import math
import re
import heapq
from typing import Dict, List, Tuple, Optional, Any, Iterable

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"

# Enforce >= 10 only
FONT_SIZES = [11, 10]

# ---- footer no-go zone (page coordinates; PyMuPDF = top-left origin) ----
NO_GO_RECT = fitz.Rect(21.00, 816.00, 411.26, 830.00)

# ---- spacing knobs ----
EDGE_PAD = 12.0
GAP_FROM_HIGHLIGHTS = 10.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 1.5

# Arrowhead
ARROW_LEN = 9.0
ARROW_HALF_WIDTH = 4.5

# For quote search robustness
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18

# ---- deterministic preference (NOT forced) ----
SIDE_LEFT_LABELS = {
    "Original source of publication.",
    "Venue is distinguished organization.",
    "Ensemble is distinguished organization.",
}
SIDE_RIGHT_LABELS = {
    "Performance date.",
    "Beneficiary lead role evidence.",
    "Highly acclaimed review of the distinguished performance.",
}

# ---- grid routing knobs ----
GRID_STEP = 8.0  # coarse grid cell size
HARD_INFLATE = (LINE_WIDTH / 2.0) + 2.0  # inflate hard obstacles so “touching” counts as collision
TEXT_SOFT_PENALTY = 45.0  # extra cost per step for moving through text area
TURN_PENALTY = 0.35       # tiny preference for fewer turns

# ---- dynamic placement knobs ----
CALLOUT_MAX_W = 180.0
CALLOUT_MIN_W = 70.0
# How far from the target union we try candidate callout centers
CANDIDATE_RADII = [30, 50, 70, 90, 120]
# Candidate directions (dx, dy) multipliers for radii
CANDIDATE_DIRS = [
    (1, 0),   (-1, 0),  (0, 1),  (0, -1),
    (1, 1),   (1, -1), (-1, 1), (-1, -1),
    (2, 1),   (2, -1), (-2, 1), (-2, -1),
    (1, 2),   (-1, 2), (1, -2), (-1, -2),
]


# ============================================================
# Geometry helpers
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad
    rr.y0 -= pad
    rr.x1 += pad
    rr.y1 += pad
    return rr


def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects:
        return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rect_is_valid(r: fitz.Rect) -> bool:
    vals = [r.x0, r.y0, r.x1, r.y1]
    return (
        all(math.isfinite(v) for v in vals)
        and (r.x1 > r.x0)
        and (r.y1 > r.y0)
    )


def _ensure_min_size(
    r: fitz.Rect,
    pr: fitz.Rect,
    min_w: float = 55.0,
    min_h: float = 14.0,
    pad: float = 2.0,
) -> fitz.Rect:
    rr = fitz.Rect(r)
    cx = (rr.x0 + rr.x1) / 2.0
    cy = (rr.y0 + rr.y1) / 2.0
    w = max(min_w, abs(rr.x1 - rr.x0))
    h = max(min_h, abs(rr.y1 - rr.y0))
    rr = fitz.Rect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    rr.x0 = max(pad, rr.x0)
    rr.y0 = max(pad, rr.y0)
    rr.x1 = min(pr.width - pad, rr.x1)
    rr.y1 = min(pr.height - pad, rr.y1)

    if rr.x1 <= rr.x0 or rr.y1 <= rr.y0:
        rr = fitz.Rect(pad, pad, pad + min_w, pad + min_h)
    return rr


# ============================================================
# Robust segment vs rect intersection (touching counts)
# Liang–Barsky line clipping
# ============================================================

def _segment_intersects_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    p = [-dx, dx, -dy, dy]
    q = [p1.x - x0, x1 - p1.x, p1.y - y0, y1 - p1.y]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return True


def _segment_hits_any(p1: fitz.Point, p2: fitz.Point, rects: Iterable[fitz.Rect]) -> bool:
    for r in rects:
        if _segment_intersects_rect(p1, p2, r):
            return True
    return False


# ============================================================
# Text area + word rects (for dynamic placement scoring)
# ============================================================

def _get_fallback_text_area(page: fitz.Page) -> fitz.Rect:
    pr = page.rect
    return fitz.Rect(pr.width * 0.12, pr.height * 0.12, pr.width * 0.88, pr.height * 0.88)


def _detect_actual_text_area(page: fitz.Page) -> fitz.Rect:
    try:
        words = page.get_text("words") or []
        if not words:
            return _get_fallback_text_area(page)
        pr = page.rect
        header_limit = pr.height * 0.12
        footer_limit = pr.height * 0.88
        x0s, x1s = [], []
        for w in words:
            x0, y0, x1, y1, text = w[:5]
            if y0 > header_limit and y1 < footer_limit and len((text or "").strip()) > 1:
                x0s.append(float(x0))
                x1s.append(float(x1))
        if not x0s:
            return _get_fallback_text_area(page)
        x0s.sort()
        x1s.sort()
        li = int(len(x0s) * 0.05)
        ri = int(len(x1s) * 0.95)
        text_left = x0s[max(0, li)]
        text_right = x1s[min(len(x1s) - 1, ri)]
        text_left = max(pr.width * 0.08, text_left)
        text_right = min(pr.width * 0.92, text_right)
        if text_right <= text_left + 50:
            return _get_fallback_text_area(page)
        return fitz.Rect(text_left, header_limit, text_right, footer_limit)
    except Exception:
        return _get_fallback_text_area(page)


def _get_word_rects(page: fitz.Page, *, inflate: float = 1.5) -> List[fitz.Rect]:
    """
    Returns many small rects representing text. Used to estimate "coverage cost"
    when placing callouts (penalize overlap with actual text).
    """
    rects: List[fitz.Rect] = []
    try:
        words = page.get_text("words") or []
        for w in words:
            x0, y0, x1, y1, t = w[:5]
            if not (t or "").strip():
                continue
            r = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
            if inflate:
                r = inflate_rect(r, inflate)
            rects.append(r)
    except Exception:
        pass
    return rects


def _rect_overlap_area(a: fitz.Rect, b: fitz.Rect) -> float:
    if not a.intersects(b):
        return 0.0
    i = a & b
    return max(0.0, (i.x1 - i.x0) * (i.y1 - i.y0))


def _overlap_cost(cand: fitz.Rect, word_rects: List[fitz.Rect]) -> float:
    """
    Cost proportional to how much actual text would be covered by the callout.
    """
    if not word_rects:
        return 0.0
    area = 0.0
    for wr in word_rects:
        area += _rect_overlap_area(cand, wr)
    # Normalize-ish: 1,000 area points ~ modest penalty, tuneable
    return area / 1000.0


# ============================================================
# Text wrapping + textbox insertion
# ============================================================

def _optimize_layout_for_callout(text: str, max_w: float) -> Tuple[int, str, float, float]:
    """
    Wrap label into <= max_w using FONT_SIZES only (>=10).
    Returns: (fontsize, wrapped_text, used_width, needed_height)
    """
    text = (text or "").strip()
    if not text:
        return 11, "", max_w, 24.0

    words = text.split()
    max_h = 240.0

    for fs in FONT_SIZES:
        usable_w = max(30.0, max_w - 10.0)
        lines: List[str] = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip() if cur else w
            if fitz.get_text_length(trial, fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        wrapped = "\n".join(lines)
        h = (len(lines) * fs * 1.25) + 10.0
        if h <= max_h or fs == FONT_SIZES[-1]:
            return fs, wrapped, max_w, h

    return FONT_SIZES[-1], text, max_w, 50.0


def _insert_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontname: str,
    fontsize: int,
    color,
    align=fitz.TEXT_ALIGN_LEFT,
    overlay: bool = True,
    max_expand_iters: int = 8,
    extra_pad_each_iter: float = 6.0,
) -> Tuple[fitz.Rect, float, int]:
    pr = page.rect
    r = _ensure_min_size(fitz.Rect(rect), pr)
    fs = int(fontsize)

    # enforce >= 10
    fs = max(10, fs)

    def attempt(rr: fitz.Rect, fsize: int) -> float:
        rr = _ensure_min_size(rr, pr)
        if not _rect_is_valid(rr):
            return -1.0
        return page.insert_textbox(
            rr, text, fontname=fontname, fontsize=fsize, color=color, align=align, overlay=overlay
        )

    ret = attempt(r, fs)
    it = 0
    while ret < 0 and it < max_expand_iters:
        need = (-ret) + extra_pad_each_iter
        r.y0 -= need / 2.0
        r.y1 += need / 2.0
        r.y0 = max(2.0, r.y0)
        r.y1 = min(pr.height - 2.0, r.y1)
        ret = attempt(r, fs)
        it += 1

    # If still doesn't fit, do NOT go below 10; just expand more.
    it = 0
    while ret < 0 and it < (max_expand_iters + 6):
        need = (-ret) + extra_pad_each_iter
        r.y0 -= need / 2.0
        r.y1 += need / 2.0
        r.y0 = max(2.0, r.y0)
        r.y1 = min(pr.height - 2.0, r.y1)
        ret = attempt(r, fs)
        it += 1

    return r, ret, fs


# ============================================================
# De-duplication
# ============================================================

def _rect_area(r: fitz.Rect) -> float:
    return max(0.0, (r.x1 - r.x0) * (r.y1 - r.y0))


def _dedupe_rects(rects: List[fitz.Rect], pad: float = 1.0) -> List[fitz.Rect]:
    if not rects:
        return []
    rr = [fitz.Rect(r) for r in rects]
    rr.sort(key=lambda r: _rect_area(r), reverse=True)
    kept: List[fitz.Rect] = []
    for r in rr:
        rbuf = inflate_rect(r, pad)
        contained = False
        for k in kept:
            if inflate_rect(k, pad).contains(rbuf):
                contained = True
                break
        if not contained:
            kept.append(r)
    kept.sort(key=lambda r: (r.y0, r.x0))
    return kept


# ============================================================
# Robust search helpers
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    t = (term or "").strip()
    if not t:
        return []
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    flags = 0
    try:
        flags |= fitz.TEXT_DEHYPHENATE
    except Exception:
        pass
    try:
        flags |= fitz.TEXT_PRESERVE_WHITESPACE
    except Exception:
        pass

    try:
        rects = page.search_for(t, flags=flags)
        if rects:
            return rects
    except Exception:
        pass

    t2 = _normalize_spaces(t)
    if t2 and t2 != t:
        try:
            rects = page.search_for(t2, flags=flags)
            if rects:
                return rects
        except Exception:
            pass

    if len(t2) >= _CHUNK:
        hits: List[fitz.Rect] = []
        step = max(10, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t2), step):
            chunk = t2[i:i + _CHUNK].strip()
            if len(chunk) < 18:
                continue
            try:
                hits.extend(page.search_for(chunk, flags=flags))
            except Exception:
                continue

        if hits:
            hits_sorted = sorted(hits, key=lambda r: (r.y0, r.x0))
            merged: List[fitz.Rect] = []
            for r in hits_sorted:
                if not merged:
                    merged.append(fitz.Rect(r))
                else:
                    last = merged[-1]
                    if last.intersects(r) or abs(last.y0 - r.y0) < 3.0:
                        merged[-1] = last | r
                    else:
                        merged.append(fitz.Rect(r))
            return merged

    return []


# ============================================================
# URL helpers
# ============================================================

def _looks_like_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("www.")


def _normalize_urlish(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.strip(" \t\r\n'\"()[]{}<>.,;")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    return s


def _is_same_urlish(a: str, b: str) -> bool:
    na = _normalize_urlish(a)
    nb = _normalize_urlish(b)
    if not na or not nb:
        return False
    return na == nb


# ============================================================
# Arrow drawing
# ============================================================

def _draw_arrowhead(page: fitz.Page, start: fitz.Point, end: fitz.Point, *, overlay: bool = True):
    vx = end.x - start.x
    vy = end.y - start.y
    d = math.hypot(vx, vy)
    if d == 0:
        return
    ux, uy = vx / d, vy / d
    bx = end.x - ux * ARROW_LEN
    by = end.y - uy * ARROW_LEN
    px = -uy
    py = ux
    p1 = fitz.Point(bx + px * ARROW_HALF_WIDTH, by + py * ARROW_HALF_WIDTH)
    p2 = fitz.Point(bx - px * ARROW_HALF_WIDTH, by - py * ARROW_HALF_WIDTH)
    tip = fitz.Point(end.x, end.y)
    page.draw_polyline([p1, tip, p2, p1], color=RED, fill=RED, width=0.0, overlay=overlay)


def _draw_line(page: fitz.Page, a: fitz.Point, b: fitz.Point, *, overlay: bool = True):
    page.draw_line(a, b, color=RED, width=LINE_WIDTH, overlay=overlay)


def _draw_poly_connector(page: fitz.Page, pts: List[fitz.Point], *, overlay: bool = True):
    if len(pts) < 2:
        return
    for a, b in zip(pts, pts[1:]):
        _draw_line(page, a, b, overlay=overlay)
    _draw_arrowhead(page, pts[-2], pts[-1], overlay=overlay)


# ============================================================
# Margin lanes (fallback)
# ============================================================

def _compute_equal_margins(page: fitz.Page) -> Tuple[fitz.Rect, fitz.Rect, fitz.Rect]:
    pr = page.rect
    text_area = _detect_actual_text_area(page)

    left_available = max(0.0, text_area.x0 - EDGE_PAD)
    right_available = max(0.0, (pr.width - EDGE_PAD) - text_area.x1)
    lane_w = max(0.0, min(left_available, right_available))
    lane_w = max(lane_w, 60.0)

    left_lane = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + lane_w, pr.height - EDGE_PAD)
    right_lane = fitz.Rect(pr.width - EDGE_PAD - lane_w, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
    return text_area, left_lane, right_lane


def _choose_side_for_label(label: str) -> str:
    if label in SIDE_LEFT_LABELS:
        return "left"
    if label in SIDE_RIGHT_LABELS:
        return "right"
    return "left"


def _rect_conflicts(r: fitz.Rect, occupied: List[fitz.Rect], pad: float = 0.0) -> bool:
    rr = inflate_rect(r, pad) if pad else r
    return any(rr.intersects(o) for o in occupied)


def _place_callout_in_lane_fallback(
    page: fitz.Page,
    lane: fitz.Rect,
    text_area: fitz.Rect,
    target_union: fitz.Rect,
    occupied_same_side: List[fitz.Rect],
    label: str,
    footer_no_go: fitz.Rect,
) -> Tuple[fitz.Rect, str, int]:
    pr = page.rect
    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)

    lane_w = lane.x1 - lane.x0
    max_w = min(CALLOUT_MAX_W, lane_w - 8.0)
    max_w = max(max_w, CALLOUT_MIN_W)

    fs, wrapped, w_used, h_needed = _optimize_layout_for_callout(label, max_w)
    w_used = min(w_used, max_w)

    def build_at_center_y(cy: float) -> fitz.Rect:
        y0 = cy - h_needed / 2.0
        y1 = cy + h_needed / 2.0
        y0 = max(lane.y0, y0)
        y1 = min(lane.y1, y1)
        if (y1 - y0) < (h_needed * 0.85):
            y1 = min(lane.y1, y0 + h_needed)
            y0 = max(lane.y0, y1 - h_needed)

        x0 = lane.x0 + 4.0
        x1 = min(lane.x1 - 4.0, x0 + w_used)
        cand = fitz.Rect(x0, y0, x1, y1)
        cand = _ensure_min_size(cand, pr, min_w=55.0, min_h=14.0)
        return cand

    def allowed(cand: fitz.Rect) -> bool:
        if not lane.contains(cand):
            return False
        if cand.intersects(text_area):
            return False
        if cand.intersects(target_no_go):
            return False
        if footer_no_go.width > 0 and footer_no_go.height > 0 and cand.intersects(footer_no_go):
            return False
        if _rect_conflicts(cand, occupied_same_side, pad=GAP_BETWEEN_CALLOUTS):
            return False
        return True

    target_y = _center(target_union).y
    scan_steps = [0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100, 140, -140, 180, -180]

    for dy in scan_steps:
        cand = build_at_center_y(target_y + dy)
        if allowed(cand):
            return cand, wrapped, fs

    y_cursor = lane.y0 + 14.0
    while y_cursor + h_needed < lane.y1 - 14.0:
        cand = build_at_center_y(y_cursor + h_needed / 2.0)
        if allowed(cand):
            return cand, wrapped, fs
        y_cursor += (h_needed + GAP_BETWEEN_CALLOUTS)

    return build_at_center_y(min(max(target_y, lane.y0 + 20), lane.y1 - 20)), wrapped, fs


# ============================================================
# Grid A* router (hard obstacles + soft text penalty)
# ============================================================

def _grid_build(
    pr: fitz.Rect,
    hard_obstacles: List[fitz.Rect],
    soft_rects: List[fitz.Rect],
    step: float,
) -> Tuple[int, int, List[List[bool]], List[List[float]]]:
    width = pr.width
    height = pr.height
    cols = max(1, int(math.ceil(width / step)))
    rows = max(1, int(math.ceil(height / step)))

    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]

    blocked = [[False for _ in range(cols)] for _ in range(rows)]
    soft_cost = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def cell_rect(i: int, j: int) -> fitz.Rect:
        x0 = i * step
        y0 = j * step
        x1 = min(width, x0 + step)
        y1 = min(height, y0 + step)
        return fitz.Rect(x0, y0, x1, y1)

    for j in range(rows):
        for i in range(cols):
            cr = cell_rect(i, j)

            if cr.x0 < EDGE_PAD or cr.y0 < EDGE_PAD or cr.x1 > (width - EDGE_PAD) or cr.y1 > (height - EDGE_PAD):
                blocked[j][i] = True
                continue

            if any(cr.intersects(h) for h in hard_inf):
                blocked[j][i] = True
                continue

            if any(cr.intersects(s) for s in soft_rects):
                soft_cost[j][i] = TEXT_SOFT_PENALTY

    return cols, rows, blocked, soft_cost


def _point_to_cell(p: fitz.Point, pr: fitz.Rect, step: float, cols: int, rows: int) -> Tuple[int, int]:
    x = _clamp(p.x, 0.0, pr.width - 1e-6)
    y = _clamp(p.y, 0.0, pr.height - 1e-6)
    i = int(x // step)
    j = int(y // step)
    i = max(0, min(cols - 1, i))
    j = max(0, min(rows - 1, j))
    return i, j


def _cell_center(i: int, j: int, pr: fitz.Rect, step: float) -> fitz.Point:
    cx = min(pr.width - 1e-6, (i * step) + (step / 2.0))
    cy = min(pr.height - 1e-6, (j * step) + (step / 2.0))
    return fitz.Point(cx, cy)


def _astar_route(
    pr: fitz.Rect,
    start: fitz.Point,
    goal: fitz.Point,
    hard_obstacles: List[fitz.Rect],
    soft_rects: List[fitz.Rect],
    step: float = GRID_STEP,
) -> List[fitz.Point]:
    cols, rows, blocked, soft_cost = _grid_build(pr, hard_obstacles, soft_rects, step)

    si, sj = _point_to_cell(start, pr, step, cols, rows)
    gi, gj = _point_to_cell(goal, pr, step, cols, rows)

    def nudge_to_free(i: int, j: int) -> Tuple[int, int]:
        if 0 <= i < cols and 0 <= j < rows and not blocked[j][i]:
            return i, j
        for rad in range(1, 10):
            for dj in range(-rad, rad + 1):
                for di in range(-rad, rad + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < cols and 0 <= jj < rows and not blocked[jj][ii]:
                        return ii, jj
        return i, j

    si, sj = nudge_to_free(si, sj)
    gi, gj = nudge_to_free(gi, gj)

    if blocked[sj][si] or blocked[gj][gi]:
        return [start, goal]

    def h(i: int, j: int) -> float:
        return math.hypot(i - gi, j - gj)

    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]

    INF = 1e18
    gscore = [[INF for _ in range(cols)] for _ in range(rows)]
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    prevdir: Dict[Tuple[int, int], Tuple[int, int]] = {}

    pq: List[Tuple[float, int, int]] = []
    gscore[sj][si] = 0.0
    heapq.heappush(pq, (h(si, sj), si, sj))

    while pq:
        f, i, j = heapq.heappop(pq)
        if (i, j) == (gi, gj):
            break

        base = gscore[j][i]
        cur_key = (i, j)
        cur_dir = prevdir.get(cur_key)

        for di, dj, w in nbrs:
            ii, jj = i + di, j + dj
            if not (0 <= ii < cols and 0 <= jj < rows):
                continue
            if blocked[jj][ii]:
                continue

            step_cost = w + soft_cost[jj][ii]
            if cur_dir is not None and (di, dj) != cur_dir:
                step_cost += TURN_PENALTY

            ng = base + step_cost
            if ng < gscore[jj][ii]:
                gscore[jj][ii] = ng
                parent[(ii, jj)] = (i, j)
                prevdir[(ii, jj)] = (di, dj)
                heapq.heappush(pq, (ng + h(ii, jj), ii, jj))

    cur = (gi, gj)
    if cur not in parent and cur != (si, sj):
        return [start, goal]

    path_cells = [cur]
    while cur != (si, sj):
        cur = parent[cur]
        path_cells.append(cur)
    path_cells.reverse()

    pts = [_cell_center(i, j, pr, step) for (i, j) in path_cells]
    pts[0] = start
    pts[-1] = goal
    return pts


def _simplify_path(pts: List[fitz.Point], hard_obstacles: List[fitz.Rect]) -> List[fitz.Point]:
    if len(pts) <= 2:
        return pts

    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]

    simplified = [pts[0]]
    i = 2
    while i < len(pts):
        a = simplified[-1]
        c = pts[i]
        if not _segment_hits_any(a, c, hard_inf):
            i += 1
            continue
        simplified.append(pts[i - 1])
        i = (i - 1) + 2

    if simplified[-1] != pts[-1]:
        simplified.append(pts[-1])

    out = [simplified[0]]
    for p in simplified[1:]:
        if math.hypot(p.x - out[-1].x, p.y - out[-1].y) >= 1.0:
            out.append(p)
    if len(out) == 1:
        out.append(pts[-1])
    return out


def _route_connector_page1_astar(
    page: fitz.Page,
    callout: fitz.Rect,
    target: fitz.Rect,
    *,
    hard_obstacles: List[fitz.Rect],
    soft_rects: List[fitz.Rect],
) -> List[fitz.Point]:
    pr = page.rect
    cc = _center(callout)

    # start just outside the callout edge facing towards the target
    tc = _center(target)
    if tc.x >= cc.x:
        start = fitz.Point(_clamp(callout.x1 + 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))
    else:
        start = fitz.Point(_clamp(callout.x0 - 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))

    # end on target edge facing callout, pulled back slightly
    if cc.x < pr.width / 2:
        end_raw = fitz.Point(target.x0, _clamp(tc.y, target.y0 + 1.0, target.y1 - 1.0))
        approach = fitz.Point(target.x0 - 3.0, end_raw.y)
    else:
        end_raw = fitz.Point(target.x1, _clamp(tc.y, target.y0 + 1.0, target.y1 - 1.0))
        approach = fitz.Point(target.x1 + 3.0, end_raw.y)

    end = _pull_back_point(approach, end_raw, ENDPOINT_PULLBACK)

    grid_pts = _astar_route(pr, start, end, hard_obstacles, soft_rects, step=GRID_STEP)
    simp = _simplify_path(grid_pts, hard_obstacles)

    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]
    for a, b in zip(simp, simp[1:]):
        if _segment_hits_any(a, b, hard_inf):
            return grid_pts

    return simp


# ============================================================
# Multi-page connector (keep your gutter routing)
# ============================================================

def _draw_multipage_connector(
    doc: fitz.Document,
    callout_page_index: int,
    callout_rect: fitz.Rect,
    target_page_index: int,
    target_rect: fitz.Rect,
    *,
    overlay: bool = True,
):
    callout_page = doc.load_page(callout_page_index)
    pr = callout_page.rect

    cc = _center(callout_rect)
    gx = pr.width - EDGE_PAD if cc.x >= pr.width / 2 else EDGE_PAD

    start_x = callout_rect.x1 if gx > pr.width / 2 else callout_rect.x0
    start = fitz.Point(start_x, _clamp(callout_rect.y1 + 2.0, EDGE_PAD, pr.height - EDGE_PAD))
    p_gutter_start = fitz.Point(gx, start.y)
    p_gutter_bottom = fitz.Point(gx, pr.height - EDGE_PAD)

    _draw_line(callout_page, start, p_gutter_start, overlay=overlay)
    _draw_line(callout_page, p_gutter_start, p_gutter_bottom, overlay=overlay)

    for pi in range(callout_page_index + 1, target_page_index):
        p = doc.load_page(pi)
        pr_i = p.rect
        gx_i = pr_i.width - EDGE_PAD if gx > pr.width / 2 else EDGE_PAD
        _draw_line(p, fitz.Point(gx_i, EDGE_PAD), fitz.Point(gx_i, pr_i.height - EDGE_PAD), overlay=overlay)

    tp = doc.load_page(target_page_index)
    pr_t = tp.rect
    gx_t = pr_t.width - EDGE_PAD if gx > pr.width / 2 else EDGE_PAD
    tc = _center(target_rect)
    y_target = _clamp(tc.y, EDGE_PAD, pr_t.height - EDGE_PAD)

    p_top = fitz.Point(gx_t, EDGE_PAD)
    p_mid = fitz.Point(gx_t, y_target)

    if gx_t > target_rect.x1:
        end_raw = fitz.Point(target_rect.x1, _clamp(y_target, target_rect.y0 + 1, target_rect.y1 - 1))
    else:
        end_raw = fitz.Point(target_rect.x0, _clamp(y_target, target_rect.y0 + 1, target_rect.y1 - 1))

    end = _pull_back_point(p_mid, end_raw, ENDPOINT_PULLBACK)

    _draw_line(tp, p_top, p_mid, overlay=overlay)
    _draw_line(tp, p_mid, end, overlay=overlay)
    _draw_arrowhead(tp, p_mid, end, overlay=overlay)


# ============================================================
# Stars helper (unchanged)
# ============================================================

_STAR_CRITERIA = {"3", "2_past", "4_past"}

def _find_high_star_tokens(page: fitz.Page) -> List[str]:
    text = page.get_text("text") or ""
    tokens: List[str] = []
    for m in re.finditer(r"(?<!\*)\*{4,5}(?!\*)", text):
        tokens.append(m.group(0))
    for m in re.finditer(r"[★☆]{5}", text):
        tok = m.group(0)
        if tok.count("★") >= 4:
            tokens.append(tok)
    out = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ============================================================
# Dynamic callout placement near target (core change)
# ============================================================

def _candidate_callout_rects_near_target(
    page: fitz.Page,
    target_union: fitz.Rect,
    *,
    w: float,
    h: float,
    prefer_side: Optional[str],
) -> List[fitz.Rect]:
    """
    Build many candidate rectangles around the target union at several radii/directions.
    prefer_side: "left"/"right"/None only biases ordering.
    """
    pr = page.rect
    tc = _center(target_union)

    # bias ordering by prefer_side
    dirs = list(CANDIDATE_DIRS)
    if prefer_side == "left":
        dirs.sort(key=lambda d: d[0])  # more negative dx first
    elif prefer_side == "right":
        dirs.sort(key=lambda d: -d[0])  # more positive dx first

    cands: List[fitz.Rect] = []
    for r in CANDIDATE_RADII:
        for dxm, dym in dirs:
            cx = tc.x + (dxm * r)
            cy = tc.y + (dym * r)

            cand = fitz.Rect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            cand = _ensure_min_size(cand, pr, min_w=55.0, min_h=14.0)

            # Keep inside a page frame
            if cand.x0 < EDGE_PAD or cand.y0 < EDGE_PAD or cand.x1 > (pr.width - EDGE_PAD) or cand.y1 > (pr.height - EDGE_PAD):
                continue

            cands.append(cand)

    return cands


def _path_stats(path: List[fitz.Point]) -> Tuple[float, int]:
    """
    Returns (length, turns).
    """
    if len(path) < 2:
        return 1e9, 999

    length = 0.0
    for a, b in zip(path, path[1:]):
        length += math.hypot(b.x - a.x, b.y - a.y)

    turns = 0
    if len(path) >= 3:
        def stepvec(p, q):
            return (round(q.x - p.x, 2), round(q.y - p.y, 2))
        prev = stepvec(path[0], path[1])
        for i in range(2, len(path)):
            cur = stepvec(path[i - 1], path[i])
            if cur != prev:
                turns += 1
            prev = cur

    return length, turns


def _place_callout_dynamic_near_target(
    page: fitz.Page,
    label: str,
    target_union: fitz.Rect,
    *,
    occupied_all: List[fitz.Rect],
    occupied_same_side: List[fitz.Rect],
    hard_obstacles_base: List[fitz.Rect],  # e.g., footer no-go + red boxes + already-placed callouts (excluding candidate)
    soft_rects: List[fitz.Rect],           # typically [text_area]
    word_rects: List[fitz.Rect],           # actual word boxes for overlap penalty
    prefer_side: Optional[str],            # "left"/"right"/None
) -> Optional[Tuple[fitz.Rect, str, int]]:
    """
    Tries placing callout near target with best score:
      - Reject overlaps with target no-go, hard obstacles, existing callouts.
      - Penalize covering text (word_rects overlap).
      - Penalize long/turny connector paths (computed by your A* router).
    """
    pr = page.rect
    footer_no_go = hard_obstacles_base[0] if hard_obstacles_base else fitz.Rect(0,0,0,0)

    # size the callout based on label and max width
    fs, wrapped, _, h_needed = _optimize_layout_for_callout(label, CALLOUT_MAX_W)
    w_used = CALLOUT_MAX_W

    # safety min/max width
    w_used = max(CALLOUT_MIN_W, min(CALLOUT_MAX_W, w_used))

    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)

    candidates = _candidate_callout_rects_near_target(
        page,
        target_union,
        w=w_used,
        h=h_needed,
        prefer_side=prefer_side,
    )
    if not candidates:
        return None

    best = None
    best_score = 1e18

    # base hard obstacles to route against; candidate itself will be appended per-candidate
    base_hard = list(hard_obstacles_base)

    for cand in candidates:
        if not _rect_is_valid(cand):
            continue

        # quick rejects
        if cand.intersects(target_no_go):
            continue
        if footer_no_go.width > 0 and footer_no_go.height > 0 and cand.intersects(footer_no_go):
            continue
        if _rect_conflicts(cand, occupied_all, pad=GAP_BETWEEN_CALLOUTS):
            continue
        # keep a bit stricter on same side stacking, but still allow if needed
        if _rect_conflicts(cand, occupied_same_side, pad=(GAP_BETWEEN_CALLOUTS / 2.0)):
            # don't reject outright; just penalize
            same_side_pen = 80.0
        else:
            same_side_pen = 0.0

        # reject if cand overlaps any hard obstacle (except footer already checked)
        hard_hit = False
        for h in base_hard:
            if h is footer_no_go:
                continue
            if cand.intersects(inflate_rect(h, 1.0)):
                hard_hit = True
                break
        if hard_hit:
            continue

        # route a connector for feasibility/score
        hard_for_route = base_hard + [cand]  # treat candidate as obstacle as well

        path = _route_connector_page1_astar(
            page,
            callout=cand,
            target=target_union,
            hard_obstacles=hard_for_route,
            soft_rects=soft_rects,
        )

        # If router fell back to straight line, validate it doesn't hit hard obstacles
        hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_for_route]
        invalid = False
        for a, b in zip(path, path[1:]):
            if _segment_hits_any(a, b, hard_inf):
                invalid = True
                break
        if invalid:
            continue

        length, turns = _path_stats(path)
        cover_cost = _overlap_cost(cand, word_rects)

        # Side bias: if prefer_side is set, discourage opposite side
        bias = 0.0
        cc = _center(cand)
        if prefer_side == "left" and cc.x > pr.width / 2:
            bias += 25.0
        if prefer_side == "right" and cc.x < pr.width / 2:
            bias += 25.0

        # Score weights: tune as needed
        score = (
            (1.0 * length) +
            (18.0 * turns) +
            (140.0 * cover_cost) +
            same_side_pen +
            bias
        )

        if score < best_score:
            best_score = score
            best = (cand, wrapped, fs)

    return best


# ============================================================
# Main entrypoint
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    """
    Pipeline:
      1) Detect text area on page 1 + gather word rects
      2) Find hits and draw red boxes (hard obstacles)
      3) Plan callouts using dynamic near-target placement, with gutters as fallback
      4) Route arrows (page 1 via A*; other pages via gutter connector)
      5) Draw arrows first
      6) Draw callout white boxes + text last
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)
    pr1 = page1.rect

    total_quote_hits = 0
    total_meta_hits = 0

    # occupied callouts (global) + per-side for light stacking preference
    occupied_all: List[fitz.Rect] = []
    occupied_left: List[fitz.Rect] = []
    occupied_right: List[fitz.Rect] = []

    # Store callouts to draw later (after arrows)
    callouts_to_draw: List[Dict[str, Any]] = []

    # Store connector jobs
    connectors_to_route: List[Dict[str, Any]] = []

    # Detect margins/lanes (for fallback only)
    text_area, left_lane, right_lane = _compute_equal_margins(page1)
    footer_no_go_p1 = fitz.Rect(NO_GO_RECT) & pr1

    # Word rects to penalize covering actual text
    word_rects_p1 = _get_word_rects(page1, inflate=1.2)

    # Collect all red-box rectangles on page 1 (hard obstacles for arrows and placement)
    page1_redboxes: List[fitz.Rect] = []

    # ------------------------------------------------------------
    # 1) Quote highlights (URL-like quote terms restricted to page 1)
    # ------------------------------------------------------------
    meta_url = (meta.get("source_url") or "").strip()

    for page in doc:
        for term in (quote_terms or []):
            t = (term or "").strip()
            if not t:
                continue
            is_url_term = _looks_like_url(t) or (meta_url and _is_same_urlish(t, meta_url))
            if is_url_term and page.number != 0:
                continue

            rects = _search_term(page, t)
            for r in _dedupe_rects(rects):
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1
                if page.number == 0:
                    page1_redboxes.append(r)

    # ------------------------------------------------------------
    # 2) Metadata hits + callout planning
    # ------------------------------------------------------------
    def _find_targets_across_doc(needle: str, *, page_indices: Optional[List[int]] = None) -> List[Tuple[int, fitz.Rect]]:
        out = []
        needle = (needle or "").strip()
        if not needle:
            return out
        indices = page_indices if page_indices is not None else list(range(doc.page_count))
        for pi in indices:
            p = doc.load_page(pi)
            try:
                rects = p.search_for(needle)
            except Exception:
                rects = []
            for r in rects:
                out.append((pi, r))
        return out

    def _plan_callout_for_label(
        label: str,
        targets_by_page: Dict[int, List[fitz.Rect]],
        preferred_rect_p1: Optional[fitz.Rect] = None,
    ):
        nonlocal occupied_all, occupied_left, occupied_right, callouts_to_draw, connectors_to_route

        if not targets_by_page:
            return

        # Determine anchor targets for placement: prefer page 1, else first page with hits
        if 0 in targets_by_page:
            anchor_targets = _dedupe_rects(targets_by_page[0])
        else:
            first_pi = sorted(targets_by_page.keys())[0]
            anchor_targets = _dedupe_rects(targets_by_page[first_pi])

        if not anchor_targets:
            return

        target_union = _union_rect(anchor_targets)

        prefer_side = _choose_side_for_label(label)  # preference only
        occupied_same_side = occupied_left if prefer_side == "left" else occupied_right

        # Dedup redboxes for hard obstacles
        page1_redboxes_dedup = _dedupe_rects(page1_redboxes, pad=0.5)

        # Hard obstacles for placement: footer + red boxes + already chosen callouts
        hard_base_for_placement: List[fitz.Rect] = []
        if footer_no_go_p1.width > 0 and footer_no_go_p1.height > 0:
            hard_base_for_placement.append(footer_no_go_p1)
        hard_base_for_placement.extend(page1_redboxes_dedup)
        hard_base_for_placement.extend(occupied_all)

        # Soft rects for router (text area)
        soft_rects_p1 = [fitz.Rect(text_area)]

        # 1) Try dynamic near-target placement
        placed = _place_callout_dynamic_near_target(
            page1,
            label,
            target_union,
            occupied_all=occupied_all,
            occupied_same_side=occupied_same_side,
            hard_obstacles_base=hard_base_for_placement,
            soft_rects=soft_rects_p1,
            word_rects=word_rects_p1,
            prefer_side=prefer_side,
        )

        # 2) Fallback to gutter lane if needed
        if placed is None:
            lane = left_lane if prefer_side == "left" else right_lane
            cand, wrapped, fs = _place_callout_in_lane_fallback(
                page1,
                lane=lane,
                text_area=text_area,
                target_union=target_union,
                occupied_same_side=occupied_same_side,
                label=label,
                footer_no_go=footer_no_go_p1,
            )
        else:
            cand, wrapped, fs = placed

        if not _rect_is_valid(cand):
            return

        # Register occupied
        occupied_all.append(cand)
        if _center(cand).x < pr1.width / 2:
            occupied_left.append(cand)
        else:
            occupied_right.append(cand)

        callouts_to_draw.append({
            "rect": cand,
            "text": wrapped,
            "fontsize": fs,
        })

        connectors_to_route.append({
            "callout_rect": cand,
            "targets_by_page": targets_by_page,
            "label": label,
            "preferred_rect_p1": preferred_rect_p1,
        })

    def _do_job(label: str, value: Optional[str], variants: List[str] = None):
        nonlocal total_meta_hits

        val_str = str(value or "").strip()
        if not val_str:
            return

        is_url = _looks_like_url(val_str) or (meta_url and _is_same_urlish(val_str, meta_url))
        indices = [0] if is_url else None  # URL only on page 1

        needles = list(dict.fromkeys([val_str] + (variants or [])))
        targets_by_page: Dict[int, List[fitz.Rect]] = {}

        for n in needles:
            for pi, r in _find_targets_across_doc(n, page_indices=indices):
                targets_by_page.setdefault(pi, []).append(r)

        if not targets_by_page:
            return

        # Draw red boxes for metadata hits
        for pi, rects in targets_by_page.items():
            p = doc.load_page(pi)
            for r in _dedupe_rects(rects):
                p.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_meta_hits += 1
                if pi == 0:
                    page1_redboxes.append(r)

        # If URL, keep only page 1 targets
        if is_url:
            targets_by_page = {0: targets_by_page.get(0, [])}
            if not targets_by_page[0]:
                return

        preferred_rect_p1 = None
        if label == "Beneficiary lead role evidence." and 0 in targets_by_page:
            rr = _dedupe_rects(targets_by_page[0])
            if rr:
                preferred_rect_p1 = min(rr, key=lambda x: (x.x0, x.y0))

        _plan_callout_for_label(label, targets_by_page, preferred_rect_p1=preferred_rect_p1)

    _do_job("Original source of publication.", meta.get("source_url"))
    _do_job("Venue is distinguished organization.", meta.get("venue_name"))
    _do_job("Ensemble is distinguished organization.", meta.get("ensemble_name"))
    _do_job("Performance date.", meta.get("performance_date"))
    _do_job("Beneficiary lead role evidence.", meta.get("beneficiary_name"), meta.get("beneficiary_variants"))

    # ------------------------------------------------------------
    # 3) Stars (criteria-based, unchanged behavior)
    # ------------------------------------------------------------
    if criterion_id in _STAR_CRITERIA:
        stars_map: Dict[int, List[fitz.Rect]] = {}
        for p in doc:
            for tok in _find_high_star_tokens(p):
                found = p.search_for(tok)
                if found:
                    stars_map.setdefault(p.number, []).extend(found)
                    for r in _dedupe_rects(found):
                        p.draw_rect(r, color=RED, width=BOX_WIDTH)
                        total_quote_hits += 1
                        if p.number == 0:
                            page1_redboxes.append(r)

        if stars_map:
            _plan_callout_for_label(
                "Highly acclaimed review of the distinguished performance.",
                stars_map
            )

    # ------------------------------------------------------------
    # 4) Route + draw connectors FIRST
    # ------------------------------------------------------------
    page1_redboxes_deduped = _dedupe_rects(page1_redboxes, pad=0.5)

    base_hard_obstacles_p1: List[fitz.Rect] = []
    if footer_no_go_p1.width > 0 and footer_no_go_p1.height > 0:
        base_hard_obstacles_p1.append(footer_no_go_p1)
    base_hard_obstacles_p1.extend(page1_redboxes_deduped)
    base_hard_obstacles_p1.extend([cd["rect"] for cd in callouts_to_draw])

    soft_rects_p1 = [fitz.Rect(text_area)]

    for item in connectors_to_route:
        fr = item["callout_rect"]
        preferred_rect_p1 = item.get("preferred_rect_p1")

        for pi, rects in item["targets_by_page"].items():
            rr = _dedupe_rects(rects)
            if not rr:
                continue

            if pi == 0:
                targets = rr
                if preferred_rect_p1 is not None:
                    pref = preferred_rect_p1
                    targets = [pref] + [r for r in rr if r is not pref]

                for r in targets:
                    # exclude the source callout from obstacles so route can exit it cleanly
                    hard_obs = [h for h in base_hard_obstacles_p1 if h is not fr]

                    pts = _route_connector_page1_astar(
                        page1,
                        callout=fr,
                        target=r,
                        hard_obstacles=hard_obs,
                        soft_rects=soft_rects_p1,
                    )
                    _draw_poly_connector(page1, pts, overlay=True)
            else:
                for r in rr:
                    _draw_multipage_connector(doc, 0, fr, pi, r, overlay=True)

    # ------------------------------------------------------------
    # 5) Draw callouts LAST (white box + text)
    # ------------------------------------------------------------
    for cd in callouts_to_draw:
        crect = cd["rect"]
        wtext = cd["text"]
        fs = cd["fontsize"]

        page1.draw_rect(crect, color=WHITE, fill=WHITE, overlay=True)
        _insert_textbox_fit(
            page1,
            crect,
            wtext,
            fontname=FONTNAME,
            fontsize=max(10, int(fs)),
            color=RED,
            overlay=True,
        )

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id
    }
