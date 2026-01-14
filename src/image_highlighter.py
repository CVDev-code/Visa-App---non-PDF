import io
import math
import re
import heapq
from typing import Dict, List, Tuple, Optional, Any, Iterable

import fitz  # PyMuPDF

from .image_text import get_ocr_words

RED = (1, 0, 0)
WHITE = (1, 1, 1)

BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"
FONT_SIZES = [11, 10]  # do not go below 10

EDGE_PAD = 12.0
GAP_FROM_HIGHLIGHTS = 10.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 1.5

ARROW_LEN = 9.0
ARROW_HALF_WIDTH = 4.5

_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18

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

GRID_STEP = 8.0
HARD_INFLATE = (LINE_WIDTH / 2.0) + 2.0
TEXT_SOFT_PENALTY = 45.0
TURN_PENALTY = 0.35

_STAR_CRITERIA = {"3", "2_past", "4_past"}


# ----------------------------
# OCR search helpers
# ----------------------------

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _tokenize(s: str) -> List[str]:
    s = _normalize_spaces(s).lower()
    # Keep simple: split on whitespace, strip punctuation edges
    toks = []
    for t in s.split():
        t = t.strip(" \t\r\n'\"()[]{}<>.,;:!?")
        if t:
            toks.append(t)
    return toks


def _ocr_find_term_rects(ocr_words: List[Dict], term: str) -> List[fitz.Rect]:
    """
    Find occurrences of `term` in OCR word stream by token matching.
    Returns list of union rects (pixel coords -> page coords assuming 1px=1pt).
    """
    t = (term or "").strip()
    if not t:
        return []
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    needle = _tokenize(t)
    if not needle:
        return []

    hay = [_tokenize(w["text"])[0] if _tokenize(w["text"]) else "" for w in ocr_words]
    rects: List[fitz.Rect] = []

    n = len(hay)
    m = len(needle)

    for i in range(0, max(0, n - m + 1)):
        if hay[i:i+m] == needle:
            xs0, ys0, xs1, ys1 = [], [], [], []
            for j in range(m):
                w = ocr_words[i + j]
                xs0.append(w["x0"])
                ys0.append(w["y0"])
                xs1.append(w["x1"])
                ys1.append(w["y1"])
            r = fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1))
            rects.append(r)

    # Fallback: chunk search if long term
    if not rects and len(needle) >= 8:
        # try sliding windows of 6 tokens
        win = 6
        for start in range(0, len(needle) - win + 1, max(1, win - 2)):
            sub = needle[start:start+win]
            for i in range(0, max(0, n - win + 1)):
                if hay[i:i+win] == sub:
                    xs0, ys0, xs1, ys1 = [], [], [], []
                    for j in range(win):
                        w = ocr_words[i + j]
                        xs0.append(w["x0"])
                        ys0.append(w["y0"])
                        xs1.append(w["x1"])
                        ys1.append(w["y1"])
                    rects.append(fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1)))

    return _dedupe_rects(rects, pad=1.0)


def _detect_text_area_from_ocr(ocr_words: List[Dict], pr: fitz.Rect) -> fitz.Rect:
    """
    Approximate the "text area" from OCR boxes (better than guessing margins).
    """
    if not ocr_words:
        return fitz.Rect(pr.width * 0.12, pr.height * 0.12, pr.width * 0.88, pr.height * 0.88)

    xs0, ys0, xs1, ys1 = [], [], [], []
    for w in ocr_words:
        xs0.append(float(w["x0"]))
        ys0.append(float(w["y0"]))
        xs1.append(float(w["x1"]))
        ys1.append(float(w["y1"]))

    # Robust trim to avoid a single outlier
    xs0.sort(); xs1.sort(); ys0.sort(); ys1.sort()
    def pick(arr, q):
        idx = int(len(arr) * q)
        idx = max(0, min(len(arr)-1, idx))
        return arr[idx]

    left = pick(xs0, 0.03)
    right = pick(xs1, 0.97)
    top = pick(ys0, 0.03)
    bottom = pick(ys1, 0.97)

    left = max(pr.width * 0.04, left)
    right = min(pr.width * 0.96, right)
    top = max(pr.height * 0.04, top)
    bottom = min(pr.height * 0.96, bottom)

    if right <= left + 50 or bottom <= top + 50:
        return fitz.Rect(pr.width * 0.12, pr.height * 0.12, pr.width * 0.88, pr.height * 0.88)

    return fitz.Rect(left, top, right, bottom)


# ----------------------------
# Geometry helpers (same as yours)
# ----------------------------

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


# ----------------------------
# Drawing helpers (same style)
# ----------------------------

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


# ----------------------------
# Callout layout (keep “lane” idea but computed from OCR text area)
# ----------------------------

def _compute_equal_margins_from_text_area(pr: fitz.Rect, text_area: fitz.Rect) -> Tuple[fitz.Rect, fitz.Rect]:
    left_available = max(0.0, text_area.x0 - EDGE_PAD)
    right_available = max(0.0, (pr.width - EDGE_PAD) - text_area.x1)
    lane_w = max(0.0, min(left_available, right_available))
    lane_w = max(lane_w, 60.0)

    left_lane = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + lane_w, pr.height - EDGE_PAD)
    right_lane = fitz.Rect(pr.width - EDGE_PAD - lane_w, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
    return left_lane, right_lane


def _choose_side_for_label(label: str) -> str:
    if label in SIDE_LEFT_LABELS:
        return "left"
    if label in SIDE_RIGHT_LABELS:
        return "right"
    return "left"


def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 11, "", box_width, 24.0

    words = text.split()
    max_h = 220.0

    for fs in FONT_SIZES:
        usable_w = max(30.0, box_width - 10.0)
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
            return fs, wrapped, box_width, h

    return 10, text, box_width, 50.0


def _insert_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontname: str,
    fontsize: int,
    color,
    overlay: bool = True,
):
    pr = page.rect
    r = _ensure_min_size(fitz.Rect(rect), pr)
    fs = max(10, int(fontsize))
    page.insert_textbox(r, text, fontname=fontname, fontsize=fs, color=color, overlay=overlay)


def _rect_conflicts(r: fitz.Rect, occupied: List[fitz.Rect], pad: float = 0.0) -> bool:
    rr = inflate_rect(r, pad) if pad else r
    return any(rr.intersects(o) for o in occupied)


def _place_callout_in_lane(
    pr: fitz.Rect,
    lane: fitz.Rect,
    text_area: fitz.Rect,
    target_union: fitz.Rect,
    occupied_same_side: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int]:
    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)

    lane_w = lane.x1 - lane.x0
    max_w = min(180.0, lane_w - 8.0)
    max_w = max(max_w, 70.0)

    fs, wrapped, w_used, h_needed = _optimize_layout_for_margin(label, max_w)
    fs = max(10, int(fs))
    w_used = min(w_used, max_w)

    def build_at_center_y(cy: float) -> fitz.Rect:
        y0 = cy - h_needed / 2.0
        y1 = cy + h_needed / 2.0
        y0 = max(lane.y0, y0)
        y1 = min(lane.y1, y1)
        x0 = lane.x0 + 4.0
        x1 = min(lane.x1 - 4.0, x0 + w_used)
        cand = fitz.Rect(x0, y0, x1, y1)
        return _ensure_min_size(cand, pr)

    def allowed(cand: fitz.Rect) -> bool:
        if not lane.contains(cand):
            return False
        if cand.intersects(text_area):
            return False
        if cand.intersects(target_no_go):
            return False
        if _rect_conflicts(cand, occupied_same_side, pad=GAP_BETWEEN_CALLOUTS):
            return False
        return True

    target_y = _center(target_union).y
    for dy in [0, 20, -20, 40, -40, 60, -60, 80, -80, 120, -120, 160, -160]:
        cand = build_at_center_y(target_y + dy)
        if allowed(cand):
            return cand, wrapped, fs

    return build_at_center_y(target_y), wrapped, fs


# ----------------------------
# A* routing (same idea)
# ----------------------------

def _grid_build(
    pr: fitz.Rect,
    hard_obstacles: List[fitz.Rect],
    soft_rects: List[fitz.Rect],
    step: float,
):
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


def _point_to_cell(p: fitz.Point, pr: fitz.Rect, step: float, cols: int, rows: int):
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


def _astar_route(pr, start, goal, hard_obstacles, soft_rects, step=GRID_STEP):
    cols, rows, blocked, soft_cost = _grid_build(pr, hard_obstacles, soft_rects, step)

    si, sj = _point_to_cell(start, pr, step, cols, rows)
    gi, gj = _point_to_cell(goal, pr, step, cols, rows)

    def nudge_to_free(i, j):
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

    def h(i, j):
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
        _, i, j = heapq.heappop(pq)
        if (i, j) == (gi, gj):
            break

        base = gscore[j][i]
        cur_dir = prevdir.get((i, j))

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

    path = [cur]
    while cur != (si, sj):
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    pts = [_cell_center(i, j, pr, step) for (i, j) in path]
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


def _route_connector(page: fitz.Page, callout: fitz.Rect, target: fitz.Rect, hard_obstacles, soft_rects):
    pr = page.rect
    cc = _center(callout)

    if cc.x < pr.width / 2:
        start = fitz.Point(_clamp(callout.x1 + 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))
    else:
        start = fitz.Point(_clamp(callout.x0 - 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))

    tc = _center(target)
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


# ----------------------------
# Stars (OCR-based)
# ----------------------------

def _ocr_find_star_rects(ocr_words: List[Dict]) -> List[fitz.Rect]:
    rects = []
    for w in ocr_words:
        t = w["text"]
        if t in {"*****", "★★★★☆", "★★★★★"}:
            if t.count("★") >= 4 or t.count("*") >= 4:
                rects.append(fitz.Rect(w["x0"], w["y0"], w["x1"], w["y1"]))
    return _dedupe_rects(rects, pad=1.0)


# ============================================================
# MAIN ENTRYPOINT (image bytes -> annotated PDF bytes)
# ============================================================

def annotate_image_bytes(
    image_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    """
    Image-only pipeline:
      - OCR once -> word boxes
      - create a 1-page PDF containing the image
      - locate quote/meta targets via OCR matching (rects)
      - place callouts in left/right lanes
      - route connectors avoiding boxes/callouts (A*)
      - draw arrows first, callouts last
    Output: a PDF (so your export/ZIP flow stays basically identical).
    """
    # Build a one-page PDF with the image as background
    pix = fitz.Pixmap(stream=image_bytes)
    w, h = pix.width, pix.height
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=image_bytes)

    pr = page.rect

    ocr_words = get_ocr_words(image_bytes)
    text_area = _detect_text_area_from_ocr(ocr_words, pr)
    left_lane, right_lane = _compute_equal_margins_from_text_area(pr, text_area)

    total_quote_hits = 0
    total_meta_hits = 0

    page_redboxes: List[fitz.Rect] = []
    callouts_to_draw: List[Dict[str, Any]] = []
    connectors_to_route: List[Dict[str, Any]] = []
    occupied_left: List[fitz.Rect] = []
    occupied_right: List[fitz.Rect] = []
    all_callouts: List[fitz.Rect] = []

    # ---- 1) quote term highlights ----
    for term in (quote_terms or []):
        rects = _ocr_find_term_rects(ocr_words, term)
        for r in rects:
            page.draw_rect(r, color=RED, width=BOX_WIDTH)
            total_quote_hits += 1
            page_redboxes.append(r)

    # ---- 2) metadata hits (find in OCR) ----
    def _do_job(label: str, value: Optional[str], variants: List[str] = None):
        nonlocal total_meta_hits
        val_str = str(value or "").strip()
        if not val_str:
            return

        needles = list(dict.fromkeys([val_str] + (variants or [])))
        found_rects: List[fitz.Rect] = []
        for n in needles:
            found_rects.extend(_ocr_find_term_rects(ocr_words, n))

        if not found_rects:
            return

        found_rects = _dedupe_rects(found_rects, pad=1.0)
        for r in found_rects:
            page.draw_rect(r, color=RED, width=BOX_WIDTH)
            total_meta_hits += 1
            page_redboxes.append(r)

        target_union = _union_rect(found_rects)

        side = _choose_side_for_label(label)
        lane = left_lane if side == "left" else right_lane
        occ = occupied_left if side == "left" else occupied_right

        crect, wtext, fs = _place_callout_in_lane(pr, lane, text_area, target_union, occ, label)

        if side == "left":
            occupied_left.append(crect)
        else:
            occupied_right.append(crect)

        all_callouts.append(crect)
        callouts_to_draw.append({"rect": crect, "text": wtext, "fontsize": fs})

        connectors_to_route.append({"callout_rect": crect, "targets": found_rects})

    _do_job("Original source of publication.", meta.get("source_url"))
    _do_job("Venue is distinguished organization.", meta.get("venue_name"))
    _do_job("Ensemble is distinguished organization.", meta.get("ensemble_name"))
    _do_job("Performance date.", meta.get("performance_date"))
    _do_job("Beneficiary lead role evidence.", meta.get("beneficiary_name"), meta.get("beneficiary_variants"))

    # ---- 3) stars criterion ----
    if criterion_id in _STAR_CRITERIA:
        stars = _ocr_find_star_rects(ocr_words)
        if stars:
            for r in stars:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1
                page_redboxes.append(r)

            # callout for stars
            label = "Highly acclaimed review of the distinguished performance."
            target_union = _union_rect(stars)
            side = _choose_side_for_label(label)
            lane = left_lane if side == "left" else right_lane
            occ = occupied_left if side == "left" else occupied_right

            crect, wtext, fs = _place_callout_in_lane(pr, lane, text_area, target_union, occ, label)
            if side == "left":
                occupied_left.append(crect)
            else:
                occupied_right.append(crect)
            all_callouts.append(crect)
            callouts_to_draw.append({"rect": crect, "text": wtext, "fontsize": fs})
            connectors_to_route.append({"callout_rect": crect, "targets": stars})

    # ---- 4) route + draw arrows first ----
    hard_obstacles = _dedupe_rects(page_redboxes, pad=0.5) + all_callouts
    soft_rects = [fitz.Rect(text_area)]

    for job in connectors_to_route:
        fr = job["callout_rect"]
        for tgt in job["targets"]:
            hard = [r for r in hard_obstacles if r is not fr]
            pts = _route_connector(page, fr, tgt, hard, soft_rects)
            _draw_poly_connector(page, pts, overlay=True)

    # ---- 5) draw callouts last ----
    for cd in callouts_to_draw:
        r = cd["rect"]
        page.draw_rect(r, color=WHITE, fill=WHITE, overlay=True)
        _insert_textbox_fit(
            page,
            r,
            cd["text"],
            fontname=FONTNAME,
            fontsize=max(10, int(cd["fontsize"])),
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
        "criterion_id": criterion_id,
        "input_type": "image",
    }
