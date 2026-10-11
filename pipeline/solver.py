"""Dimension-constrained refinement of a plan JSON  (your research contribution).

Idea: detection gives walls in the right place but with slightly wrong lengths.
OCR reads dimensions written on the plan (e.g. "5.00 m"). We move the wall
endpoints by robust least squares so wall lengths match the written values,
while staying close to what was detected and keeping walls axis-aligned.
Every wall that had to move is marked source = "inferred" (yellow in the 3D viewer).

Two steps, used separately in the ablation table:
    plan = snap_manhattan(plan)                  # "+ Manhattan snapping"
    plan, report = solve_dimensions(plan, cons)  # "+ Dimension solver"

constraints = [{"wall": "w1", "length": 5.0}, ...]    (metres, from OCR)

Usage:  python pipeline/solver.py plan.json constraints.json out.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


# ---------- graph helpers ----------
def _build_nodes(plan, tol=0.15):
    """Merge wall endpoints closer than tol (m) into shared nodes.
    Returns node coordinates (N,2) and, per wall, (node_i, node_j)."""
    nodes, idx = [], []
    for w in plan["walls"]:
        pair = []
        for pt in (w["p1"], w["p2"]):
            pt = np.array(pt, float)
            for k, n in enumerate(nodes):
                if np.hypot(*(n - pt)) < tol:
                    pair.append(k)
                    break
            else:
                nodes.append(pt)
                pair.append(len(nodes) - 1)
        idx.append(tuple(pair))
    return np.array(nodes), idx


def _wall_len(w):
    return float(np.hypot(w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]))


def _orientation(w, max_deg=8.0):
    """'h', 'v' or None (diagonal) for a wall."""
    dx, dy = w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]
    ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
    if ang <= max_deg:
        return "h"
    if ang >= 90 - max_deg:
        return "v"
    return None


def _update_rooms(plan, old_nodes, new_nodes, tol=0.3):
    """Move room-polygon vertices together with the wall nodes they sit on."""
    for room in plan.get("rooms", []):
        out = []
        for x, y in room["polygon"]:
            d = np.hypot(old_nodes[:, 0] - x, old_nodes[:, 1] - y)
            k = int(np.argmin(d))
            out.append(list(map(float, new_nodes[k])) if d[k] < tol else [x, y])
        room["polygon"] = out


# ---------- step 1: Manhattan snapping ----------
def snap_manhattan(plan, max_deg=8.0, node_tol=0.15):
    """Make near-horizontal / near-vertical walls exactly horizontal / vertical."""
    plan = copy.deepcopy(plan)
    nodes, idx = _build_nodes(plan, node_tol)
    old = nodes.copy()
    # average the coordinate shared by walls that should be aligned
    for _ in range(3):
        for w, (i, j) in zip(plan["walls"], idx):
            o = _orientation({"p1": nodes[i], "p2": nodes[j]}, max_deg)
            if o == "h":
                y = (nodes[i][1] + nodes[j][1]) / 2
                nodes[i][1] = nodes[j][1] = y
            elif o == "v":
                x = (nodes[i][0] + nodes[j][0]) / 2
                nodes[i][0] = nodes[j][0] = x
    _apply_nodes(plan, nodes, idx)
    _update_rooms(plan, old, nodes)
    return plan


def _apply_nodes(plan, nodes, idx):
    for w, (i, j) in zip(plan["walls"], idx):
        w["p1"], w["p2"] = nodes[i].tolist(), nodes[j].tolist()


# ---------- step 2: dimension solver ----------
def solve_dimensions(plan, constraints, node_tol=0.15, max_rel_dev=0.4,
                     reg=0.05, f_scale=0.05):
    """Robust least squares on wall endpoints.
    constraints: [{"wall": id, "length": metres}]
    max_rel_dev: a reading further than this fraction from the detected length
                 is treated as an OCR error and ignored.
    Returns (new_plan, report)."""
    plan = copy.deepcopy(plan)
    walls = {w["id"]: k for k, w in enumerate(plan["walls"])}
    nodes0, idx = _build_nodes(plan, node_tol)
    old_lengths = [_wall_len(w) for w in plan["walls"]]
    orient = [_orientation(w) for w in plan["walls"]]

    used, rejected = [], []
    for c in constraints:
        k = walls.get(c["wall"])
        if k is None:
            rejected.append({**c, "reason": "unknown wall"})
            continue
        dev = abs(c["length"] - old_lengths[k]) / max(old_lengths[k], 1e-6)
        if dev > max_rel_dev:
            rejected.append({**c, "reason": f"{dev:.0%} away from detected length (likely OCR error)"})
        else:
            used.append((k, float(c["length"])))

    def residuals(v):
        n = v.reshape(-1, 2)
        r = []
        for k, target in used:                      # written dimensions
            i, j = idx[k]
            r.append(np.hypot(*(n[j] - n[i])) - target)
        for k, (i, j) in enumerate(idx):            # keep walls straight
            if orient[k] == "h":
                r.append(n[j][1] - n[i][1])
            elif orient[k] == "v":
                r.append(n[j][0] - n[i][0])
        r.extend(reg * (n - nodes0).ravel())        # stay near the detection
        return np.array(r)

    sol = least_squares(residuals, nodes0.ravel(), loss="soft_l1", f_scale=f_scale)
    nodes1 = sol.x.reshape(-1, 2)

    _apply_nodes(plan, nodes1, idx)
    _update_rooms(plan, nodes0, nodes1)

    moved = []
    for k, w in enumerate(plan["walls"]):
        new_len = _wall_len(w)
        if abs(new_len - old_lengths[k]) > 0.01:     # changed by more than 1 cm
            w["source"] = "inferred"
            moved.append(w["id"])
        # keep openings at the same relative place on the wall
        scale = new_len / old_lengths[k] if old_lengths[k] > 1e-6 else 1.0
        for key in ("doors", "windows"):
            for o in plan.get(key, []):
                if o["wall"] == w["id"]:
                    o["pos"] = o["pos"] * scale
    report = {"used": len(used), "rejected": rejected, "moved_walls": moved,
              "cost": float(sol.cost)}
    return plan, report


# =====================================================================================
# v2: "axis-line" solver  (works on real plans: thick walls, pieces that do not share endpoints)
#
# Real plans are mostly made of straight lines at x = const or y = const. Instead of merging wall
# endpoints (which breaks when walls are thick), we group ALL coordinates (wall ends + room corners)
# into shared lines per axis, then solve for the position of those lines. A corrected line moves
# everything attached to it, so one dimension fixes walls AND room corners.
# =====================================================================================
def _refs(plan):
    """references to every [x, y] point in the plan: wall ends and room corners"""
    refs = []
    for w in plan["walls"]:
        refs += [(w, "p1"), (w, "p2")]
    for r in plan.get("rooms", []):
        refs += [(r["polygon"], i) for i in range(len(r["polygon"]))]
    return refs


def _cluster(vals, tol):
    """1-D single-linkage clustering: values closer than tol share a line"""
    vals = np.asarray(vals, float)
    lab = np.zeros(len(vals), int)
    if len(vals) == 0:
        return lab
    order = np.argsort(vals)
    cur = 0
    for a, b in zip(order[:-1], order[1:]):
        if vals[b] - vals[a] > tol:
            cur += 1
        lab[b] = cur
    return lab


def snap_lines(plan, tol=0.02, tilt_deg=3.0):
    """Conservative clean-up: straighten almost-axis-aligned walls and merge coordinates that differ
    by less than `tol` (2 cm). Leaves already-clean plans untouched."""
    plan = copy.deepcopy(plan)
    for w in plan["walls"]:
        dx, dy = w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if 1e-6 < ang < tilt_deg:                      # nearly horizontal
            y = (w["p1"][1] + w["p2"][1]) / 2
            w["p1"][1] = w["p2"][1] = y
        elif 90 - tilt_deg < ang < 90 - 1e-6:          # nearly vertical
            x = (w["p1"][0] + w["p2"][0]) / 2
            w["p1"][0] = w["p2"][0] = x
    refs = _refs(plan)
    for axis in (0, 1):
        vals = [c[k][axis] for c, k in refs]
        lab = _cluster(vals, tol)
        for l in np.unique(lab):
            idx = np.where(lab == l)[0]
            m = float(np.mean([vals[i] for i in idx]))
            for i in idx:
                c, k = refs[i]
                c[k] = list(c[k])
                c[k][axis] = m
    return plan


def calibrate_scale(plan, constraints, items=None, min_labels=3):
    """If the plan was converted to metres with the wrong pixels-per-metre, every wall is too long or too
    short by the same factor. The written dimensions reveal it: scale = median(written / measured).
    Returns (plan, factor, items) with plan (and OCR item positions) rescaled."""
    walls = {w["id"]: w for w in plan["walls"]}
    ratios = []
    for c in constraints:
        w = walls.get(c["wall"])
        if w is not None and _wall_len(w) > 0.3:
            ratios.append(c["length"] / _wall_len(w))
    if len(ratios) < min_labels:
        return plan, 1.0, items
    f = float(np.median(ratios))
    agree = float(np.mean(np.abs(np.array(ratios) / f - 1) < 0.04))   # share of labels that agree on this factor
    if abs(f - 1) < 0.03 or not (0.5 < f < 2.0) or agree < 0.6:       # tiny or inconsistent -> just noise, do nothing
        return plan, 1.0, items
    plan = copy.deepcopy(plan)
    for w in plan["walls"]:
        w["p1"] = [w["p1"][0] * f, w["p1"][1] * f]
        w["p2"] = [w["p2"][0] * f, w["p2"][1] * f]
        if "thickness" in w:
            w["thickness"] *= f
    for r in plan.get("rooms", []):
        r["polygon"] = [[x * f, y * f] for x, y in r["polygon"]]
    for key in ("doors", "windows"):
        for o in plan.get(key, []):
            o["pos"] *= f
            o["width"] = o.get("width", 0.9) * f
    if items is not None:
        items = [{**it, "center": [it["center"][0] * f, it["center"][1] * f]} for it in items]
    return plan, f, items


def solve_lines(plan, constraints, tol=0.05, max_rel_dev=0.4, reg=0.05, f_scale=0.05, mark_tol=0.05):
    """Robust least squares on the positions of shared x- and y-lines.
    constraints: [{"wall": id, "length": metres}]. Readings further than max_rel_dev from the detected
    length are ignored as OCR errors. A wall that moved or changed length by more than mark_tol (5 cm) is
    marked source = "inferred" (yellow in the viewer); smaller changes stay "detected"."""
    plan = copy.deepcopy(plan)
    old_pts = [{"p1": list(w["p1"]), "p2": list(w["p2"])} for w in plan["walls"]]
    refs = _refs(plan)
    wall_idx = {w["id"]: k for k, w in enumerate(plan["walls"])}
    old_len = [_wall_len(w) for w in plan["walls"]]
    n = len(refs)
    vals = [np.array([c[k][0] for c, k in refs]), np.array([c[k][1] for c, k in refs])]
    lab = [_cluster(vals[0], tol), _cluster(vals[1], tol)]
    pos0 = [np.array([vals[a][lab[a] == l].mean() for l in range(lab[a].max() + 1)]) for a in (0, 1)]

    cons, rejected = [[], []], []
    for c in constraints:
        k = wall_idx.get(c["wall"])
        if k is None:
            rejected.append({**c, "reason": "unknown wall"})
            continue
        w = plan["walls"][k]
        dx, dy = w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        axis = 0 if ang < 8 else 1 if ang > 82 else None
        if axis is None:
            rejected.append({**c, "reason": "wall is diagonal"})
            continue
        dev = abs(c["length"] - old_len[k]) / max(old_len[k], 1e-6)
        if dev > max_rel_dev:
            rejected.append({**c, "reason": f"{dev:.0%} away from detected length (likely OCR error)"})
            continue
        a, b = lab[axis][2 * k], lab[axis][2 * k + 1]
        if a == b:
            continue
        sign = 1.0 if (dx if axis == 0 else dy) > 0 else -1.0
        cons[axis].append((a, b, sign, float(c["length"])))

    new_pos = []
    for a in (0, 1):
        def res(p, a=a):
            r = [s * (p[j] - p[i]) - L for i, j, s, L in cons[a]]
            r += list(reg * (p - pos0[a]))
            return np.array(r)
        sol = least_squares(res, pos0[a], loss="soft_l1", f_scale=f_scale)
        new_pos.append(sol.x)
    for a in (0, 1):
        for i, (c, k) in enumerate(refs):
            c[k] = list(c[k])
            c[k][a] = float(new_pos[a][lab[a][i]])

    moved = []
    for k, w in enumerate(plan["walls"]):
        nl = _wall_len(w)
        shift = max(float(np.hypot(w[q][0] - old_pts[k][q][0], w[q][1] - old_pts[k][q][1])) for q in ("p1", "p2"))
        if abs(nl - old_len[k]) > mark_tol or shift > mark_tol:   # length change OR position change
            w["source"] = "inferred"
            moved.append(w["id"])
        scale = nl / old_len[k] if old_len[k] > 1e-6 else 1.0
        for key in ("doors", "windows"):
            for o in plan.get(key, []):
                if o["wall"] == w["id"]:
                    o["pos"] *= scale
    report = {"used": sum(len(c) for c in cons), "rejected": rejected, "moved_walls": moved}
    return plan, report


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cons = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    plan = snap_manhattan(plan)
    plan, report = solve_dimensions(plan, cons)
    Path(sys.argv[3]).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
