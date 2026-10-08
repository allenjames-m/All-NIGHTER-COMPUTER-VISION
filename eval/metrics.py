"""Scoring for plan JSONs (same format as pipeline/build_3d.py).

Usage (from the repo root):
    python eval/metrics.py data/gt_plan.json data/pred_plan.json
or in code:
    from eval.metrics import evaluate
    scores = evaluate(gt_plan_dict, pred_plan_dict)

Metrics
  layout_iou      overlap / union of all room polygons (1.0 = perfect)
  room_iou_mean   mean IoU after matching each GT room to its best predicted room
  dim_error_cm    mean absolute error (cm) of room width/height (bounding box)
                  and of wall lengths matched by midpoint proximity
  door_f1/window_f1  openings match if their centres are within `tol` metres
  completeness    fraction of GT rooms / doors / windows that were found
"""
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


# ---------- helpers ----------
def _poly(points):
    p = Polygon(points)
    return p if p.is_valid else p.buffer(0)


def _rooms(plan):
    return [_poly(r["polygon"]) for r in plan.get("rooms", []) if len(r.get("polygon", [])) >= 3]


def _wall_lookup(plan):
    return {w["id"]: w for w in plan.get("walls", [])}


def _opening_centres(plan, key):
    """World (x,y) centre of every door/window."""
    walls = _wall_lookup(plan)
    pts = []
    for o in plan.get(key, []):
        w = walls.get(o.get("wall"))
        if w is None:
            continue
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L < 1e-9:
            continue
        t = o["pos"] / L
        pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return pts


def _f1(gt_pts, pred_pts, tol):
    """Greedy one-to-one matching of points within tol metres."""
    if not gt_pts and not pred_pts:
        return 1.0, 1.0, 1.0
    if not gt_pts or not pred_pts:
        return 0.0, 0.0, 0.0
    pairs = sorted(
        ((np.hypot(g[0] - p[0], g[1] - p[1]), i, j)
         for i, g in enumerate(gt_pts) for j, p in enumerate(pred_pts)))
    used_g, used_p, tp = set(), set(), 0
    for d, i, j in pairs:
        if d > tol:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        tp += 1
    prec, rec = tp / len(pred_pts), tp / len(gt_pts)
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


# ---------- metrics ----------
def layout_iou(gt_plan, pred_plan):
    g, p = _rooms(gt_plan), _rooms(pred_plan)
    if not g or not p:
        return 0.0
    gu, pu = unary_union(g), unary_union(p)
    union = gu.union(pu).area
    return float(gu.intersection(pu).area / union) if union > 0 else 0.0


def room_iou_mean(gt_plan, pred_plan):
    g, p = _rooms(gt_plan), _rooms(pred_plan)
    if not g:
        return 0.0
    scores = []
    for gr in g:
        best = 0.0
        for pr in p:
            u = gr.union(pr).area
            if u > 0:
                best = max(best, gr.intersection(pr).area / u)
        scores.append(best)
    return float(np.mean(scores))


def dim_error_cm(gt_plan, pred_plan):
    """Mean abs error in cm over (a) room bbox width/height and (b) wall lengths."""
    errs = []
    # (a) rooms matched by best IoU
    g, p = _rooms(gt_plan), _rooms(pred_plan)
    for gr in g:
        best, best_pr = 0.0, None
        for pr in p:
            u = gr.union(pr).area
            iou = gr.intersection(pr).area / u if u > 0 else 0
            if iou > best:
                best, best_pr = iou, pr
        if best_pr is None:
            continue
        gx0, gy0, gx1, gy1 = gr.bounds
        px0, py0, px1, py1 = best_pr.bounds
        errs += [abs((gx1 - gx0) - (px1 - px0)), abs((gy1 - gy0) - (py1 - py0))]
    # (b) walls matched by nearest midpoint
    def mids(plan):
        out = []
        for w in plan.get("walls", []):
            (x1, y1), (x2, y2) = w["p1"], w["p2"]
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(np.hypot(x2 - x1, y2 - y1))))
        return out
    gm, pm = mids(gt_plan), mids(pred_plan)
    for gx, gy, gl in gm:
        if not pm:
            break
        d = [np.hypot(gx - px, gy - py) for px, py, _ in pm]
        j = int(np.argmin(d))
        if d[j] < 0.5:  # only compare walls that clearly are the same wall
            errs.append(abs(gl - pm[j][2]))
    return float(np.mean(errs) * 100) if errs else float("nan")


def evaluate(gt_plan, pred_plan, tol=0.5):
    dg, dp = _opening_centres(gt_plan, "doors"), _opening_centres(pred_plan, "doors")
    wg, wp = _opening_centres(gt_plan, "windows"), _opening_centres(pred_plan, "windows")
    d_f1, _, d_rec = _f1(dg, dp, tol)
    w_f1, _, w_rec = _f1(wg, wp, tol)
    # completeness: room recall at IoU >= 0.5
    g, p = _rooms(gt_plan), _rooms(pred_plan)
    found = 0
    for gr in g:
        for pr in p:
            u = gr.union(pr).area
            if u > 0 and gr.intersection(pr).area / u >= 0.5:
                found += 1
                break
    room_rec = found / len(g) if g else 1.0
    return {
        "layout_iou": round(layout_iou(gt_plan, pred_plan), 4),
        "room_iou_mean": round(room_iou_mean(gt_plan, pred_plan), 4),
        "dim_error_cm": round(dim_error_cm(gt_plan, pred_plan), 2),
        "door_f1": round(d_f1, 4),
        "window_f1": round(w_f1, 4),
        "completeness": round(float(np.mean([room_rec, d_rec, w_rec])), 4),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    gt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    pred = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    for k, v in evaluate(gt, pred).items():
        print(f"{k:15s} {v}")


if __name__ == "__main__":
    main()
