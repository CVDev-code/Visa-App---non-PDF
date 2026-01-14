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
FONT_SIZES = [11, 10, 9, 8]

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

# ---- deterministic side assignment ----
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
GRID_STEP = 8.0              # coarse grid cell size (6–10pt recommended)
HARD_INFLATE = (LINE_WIDTH / 2.0) + 2.0  # inflate hard obstacles so “touching” counts as collision
TEXT_SOFT_PENALTY = 45.0     # extra cost per step for moving through text area
TURN_PENALTY = 0.35          # tiny preference for fewer turns (optional)


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
    # Returns True if segment intersects/touches the rect.
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
    return True  # intersects or touches


def _segment_hits_any(p1: fitz.Point, p2: fitz.Point, rects: Iterable[fitz.Rect]) -> bool:
    for r in rects:
        if _segment_intersects_rect(p1, p2, r):
            return True
    return False


# ============================================================
# Text area detection (dynamic margins)
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


# ============================================================
# Text wrapping + textbox insertion
# ============================================================

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

    return FONT_SIZES[-1], text, box_width, 50.0


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

    shrink_tries = 0
    while ret < 0 and fs > FONT_SIZES[-1] and shrink_tries < 4:
        fs -= 1
        r = _ensure_min_size(fitz.Rect(rect), pr)
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
        shrink_tries += 1

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
# Arrow drawing (same style)
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
# Margin lanes (equal width) — keep the same logic
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


def _place_callout_in_lane(
    page: fitz.Page,
    lane: fitz.Rect,
    text_area: fitz.Rect,
    target_union: fitz.Rect,
    occupied_same_side: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int]:
    pr = page.rect
    footer_no_go = fitz.Rect(NO_GO_RECT) & pr
    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)

    lane_w = lane.x1 - lane.x0
    max_w = min(180.0, lane_w - 8.0)
    max_w = max(max_w, 70.0)

    fs, wrapped, w_used, h_needed = _optimize_layout_for_margin(label, max_w)
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

    # Inflate hard obstacles so “touching” counts
    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]

    blocked = [[False for _ in range(cols)] for _ in range(rows)]
    soft_cost = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def cell_rect(i: int, j: int) -> fitz.Rect:
        x0 = i * step
        y0 = j * step
        x1 = min(width, x0 + step)
        y1 = min(height, y0 + step)
        return fitz.Rect(x0, y0, x1, y1)

    # Fill grid
    for j in range(rows):
        for i in range(cols):
            cr = cell_rect(i, j)

            # stay inside EDGE_PAD boundary (acts like a page frame)
            if cr.x0 < EDGE_PAD or cr.y0 < EDGE_PAD or cr.x1 > (width - EDGE_PAD) or cr.y1 > (height - EDGE_PAD):
                blocked[j][i] = True
                continue

            # hard obstacles
            if any(cr.intersects(h) for h in hard_inf):
                blocked[j][i] = True
                continue

            # soft obstacles (text area)
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

    # If start/goal fall in blocked cells, nudge to nearest unblocked within a small radius
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
        return [start, goal]  # hard fallback

    def h(i: int, j: int) -> float:
        return math.hypot(i - gi, j - gj)

    # 8-neighborhood
    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]

    INF = 1e18
    gscore = [[INF for _ in range(cols)] for _ in range(rows)]
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # store prev direction for slight turn penalty
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
            # tiny turn penalty
            if cur_dir is not None and (di, dj) != cur_dir:
                step_cost += TURN_PENALTY

            ng = base + step_cost
            if ng < gscore[jj][ii]:
                gscore[jj][ii] = ng
                parent[(ii, jj)] = (i, j)
                prevdir[(ii, jj)] = (di, dj)
                heapq.heappush(pq, (ng + h(ii, jj), ii, jj))

    # reconstruct
    cur = (gi, gj)
    if cur not in parent and cur != (si, sj):
        return [start, goal]  # fallback if no path found

    path_cells = [cur]
    while cur != (si, sj):
        cur = parent[cur]
        path_cells.append(cur)
    path_cells.reverse()

    pts = [_cell_center(i, j, pr, step) for (i, j) in path_cells]
    pts[0] = start
    pts[-1] = goal
    return pts


def _simplify_path(
    pts: List[fitz.Point],
    hard_obstacles: List[fitz.Rect],
) -> List[fitz.Point]:
    if len(pts) <= 2:
        return pts

    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]

    simplified = [pts[0]]
    anchor_idx = 0
    i = 2
    while i < len(pts):
        a = simplified[-1]
        c = pts[i]
        # if we can see c directly from a without hitting hard obstacles, keep stretching
        if not _segment_hits_any(a, c, hard_inf):
            i += 1
            continue
        # otherwise, lock in the previous point
        simplified.append(pts[i - 1])
        anchor_idx = i - 1
        i = anchor_idx + 2

    if simplified[-1] != pts[-1]:
        simplified.append(pts[-1])

    # remove near-duplicates / tiny segments
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

    # start just outside the callout, from the edge facing the text area
    # (left-lane callout -> exit from right edge; right-lane callout -> exit from left edge)
    if cc.x < pr.width / 2:
        start = fitz.Point(_clamp(callout.x1 + 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))
    else:
        start = fitz.Point(_clamp(callout.x0 - 2.0, EDGE_PAD, pr.width - EDGE_PAD), _clamp(cc.y, EDGE_PAD, pr.height - EDGE_PAD))

    # end on the target edge facing the callout side, pulled back so arrowhead doesn't sit on the box
    tc = _center(target)
    if cc.x < pr.width / 2:
        end_raw = fitz.Point(target.x0, _clamp(tc.y, target.y0 + 1.0, target.y1 - 1.0))
        approach = fitz.Point(target.x0 - 3.0, end_raw.y)
    else:
        end_raw = fitz.Point(target.x1, _clamp(tc.y, target.y0 + 1.0, target.y1 - 1.0))
        approach = fitz.Point(target.x1 + 3.0, end_raw.y)

    end = _pull_back_point(approach, end_raw, ENDPOINT_PULLBACK)

    # Route on grid then simplify
    grid_pts = _astar_route(pr, start, end, hard_obstacles, soft_rects, step=GRID_STEP)
    simp = _simplify_path(grid_pts, hard_obstacles)

    # Safety: ensure simplified segments still don't hit hard obstacles (rare edge)
    hard_inf = [inflate_rect(r, HARD_INFLATE) for r in hard_obstacles]
    for a, b in zip(simp, simp[1:]):
        if _segment_hits_any(a, b, hard_inf):
            return grid_pts  # fall back to unsimplified

    return simp


# ============================================================
# Multi-page connector (unchanged gutter routing)
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

    # pick gutter by callout position
    cc = _center(callout_rect)
    gx = pr.width - EDGE_PAD if cc.x >= pr.width / 2 else EDGE_PAD

    # start outside callout (bottom)
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
# Stars helper
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
# Main entrypoint
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    """
    Pipeline (amended):
      1) Detect text area
      2) Find hits and draw red boxes (obstacles)
      3) Define gutters/lanes + footer no-go (hard)
      4) Place callout rects in gutter (geometry constraints), BUT delay drawing
      5) Route all arrows with A* (hard obstacles = callouts + red boxes + no-go; soft = text area)
      6) Draw arrows first
      7) Draw callout white boxes + text last (so arrows never appear over callouts)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)
    pr1 = page1.rect

    total_quote_hits = 0
    total_meta_hits = 0

    occupied_left: List[fitz.Rect] = []
    occupied_right: List[fitz.Rect] = []
    all_callouts: List[fitz.Rect] = []

    # Store callouts to draw later (after arrows)
    callouts_to_draw: List[Dict[str, Any]] = []

    # Store connector jobs; each job can create multiple arrows (one per red box on page 1)
    connectors_to_route: List[Dict[str, Any]] = []

    # Detect margins/lanes (keep same space/coords logic)
    text_area, left_lane, right_lane = _compute_equal_margins(page1)
    footer_no_go_p1 = fitz.Rect(NO_GO_RECT) & pr1

    # Collect all red-box rectangles on page 1 (hard obstacles for arrows)
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
    # 2) Metadata hits + callout planning (place callouts now, draw later)
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

    def _plan_and_register_callout(
        label: str,
        targets_by_page: Dict[int, List[fitz.Rect]],
        preferred_rect_p1: Optional[fitz.Rect] = None,
    ):
        nonlocal occupied_left, occupied_right, all_callouts, callouts_to_draw, connectors_to_route

        if not targets_by_page:
            return

        if 0 in targets_by_page:
            anchor_targets = _dedupe_rects(targets_by_page[0])
        else:
            first_pi = sorted(targets_by_page.keys())[0]
            anchor_targets = _dedupe_rects(targets_by_page[first_pi])

        if not anchor_targets:
            return

        target_union = _union_rect(anchor_targets)
        side = _choose_side_for_label(label)
        lane = left_lane if side == "left" else right_lane
        occupied = occupied_left if side == "left" else occupied_right

        crect, wtext, fs = _place_callout_in_lane(
            page1,
            lane=lane,
            text_area=text_area,
            target_union=target_union,
            occupied_same_side=occupied,
            label=label,
        )

        if footer_no_go_p1.width > 0 and footer_no_go_p1.height > 0 and crect.intersects(footer_no_go_p1):
            shift = (crect.y1 - footer_no_go_p1.y0) + EDGE_PAD
            crect = fitz.Rect(crect.x0, crect.y0 - shift, crect.x1, crect.y1 - shift)
            crect = _ensure_min_size(crect, pr1)

        if not _rect_is_valid(crect):
            return

        # Reserve space (so future callouts avoid this)
        if side == "left":
            occupied_left.append(crect)
        else:
            occupied_right.append(crect)

        all_callouts.append(crect)

        # Store for later drawing (after arrows)
        callouts_to_draw.append({
            "rect": crect,
            "text": wtext,
            "fontsize": fs,
        })

        # Store routing jobs (can generate multiple arrows)
        connectors_to_route.append({
            "callout_rect": crect,
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

        for pi, rects in targets_by_page.items():
            p = doc.load_page(pi)
            for r in _dedupe_rects(rects):
                p.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_meta_hits += 1
                if pi == 0:
                    page1_redboxes.append(r)

        if is_url:
            targets_by_page = {0: targets_by_page.get(0, [])}
            if not targets_by_page[0]:
                return

        preferred_rect_p1 = None
        if label == "Beneficiary lead role evidence." and 0 in targets_by_page:
            rr = _dedupe_rects(targets_by_page[0])
            if rr:
                preferred_rect_p1 = min(rr, key=lambda x: (x.x0, x.y0))

        _plan_and_register_callout(label, targets_by_page, preferred_rect_p1=preferred_rect_p1)

    _do_job("Original source of publication.", meta.get("source_url"))
    _do_job("Venue is distinguished organization.", meta.get("venue_name"))
    _do_job("Ensemble is distinguished organization.", meta.get("ensemble_name"))
    _do_job("Performance date.", meta.get("performance_date"))
    _do_job("Beneficiary lead role evidence.", meta.get("beneficiary_name"), meta.get("beneficiary_variants"))

    # ------------------------------------------------------------
    # 3) Stars
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
            _plan_and_register_callout(
                "Highly acclaimed review of the distinguished performance.",
                stars_map
            )

    # ------------------------------------------------------------
    # 4) Route + draw connectors FIRST
    #     Hard obstacles: callouts + footer no-go + red boxes
    #     Soft obstacles: text area (penalty)
    # ------------------------------------------------------------
    page1_redboxes_deduped = _dedupe_rects(page1_redboxes, pad=0.5)

    # Fixed hard obstacles on page 1
    base_hard_obstacles_p1: List[fitz.Rect] = []
    if footer_no_go_p1.width > 0 and footer_no_go_p1.height > 0:
        base_hard_obstacles_p1.append(footer_no_go_p1)
    base_hard_obstacles_p1.extend(page1_redboxes_deduped)
    base_hard_obstacles_p1.extend(all_callouts)

    # Soft rects (text area)
    soft_rects_p1 = [fitz.Rect(text_area)]

    for item in connectors_to_route:
        fr = item["callout_rect"]
        preferred_rect_p1 = item.get("preferred_rect_p1")

        for pi, rects in item["targets_by_page"].items():
            rr = _dedupe_rects(rects)
            if not rr:
                continue

            if pi == 0:
                # Allow multiple arrows: connect ALL red boxes on page 1 for this label,
                # unless a preferred is supplied (in which case include it first but still draw all)
                targets = rr
                if preferred_rect_p1 is not None:
                    # Put preferred first, but keep the rest too
                    pref = preferred_rect_p1
                    targets = [pref] + [r for r in rr if r is not pref]

                for r in targets:
                    # Build hard obstacle list; exclude the source callout so the route can exit it,
                    # but keep all other callouts blocked.
                    hard_obs = list(base_hard_obstacles_p1)
                    hard_obs.extend([c for c in all_callouts if c is not fr])

                    pts = _route_connector_page1_astar(
                        page1,
                        callout=fr,
                        target=r,
                        hard_obstacles=hard_obs,
                        soft_rects=soft_rects_p1,
                    )
                    _draw_poly_connector(page1, pts, overlay=True)  # arrows drawn first
            else:
                for r in rr:
                    _draw_multipage_connector(doc, 0, fr, pi, r, overlay=True)

    # ------------------------------------------------------------
    # 5) Draw callouts LAST (white box + text), so arrows never appear on top
    # ------------------------------------------------------------
    for cd in callouts_to_draw:
        crect = cd["rect"]
        wtext = cd["text"]
        fs = cd["fontsize"]

        # Cover underlying arrows
        page1.draw_rect(crect, color=WHITE, fill=WHITE, overlay=True)

        # Then draw the text (same font/size logic)
        _insert_textbox_fit(
            page1,
            crect,
            wtext,
            fontname=FONTNAME,
            fontsize=fs,
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
