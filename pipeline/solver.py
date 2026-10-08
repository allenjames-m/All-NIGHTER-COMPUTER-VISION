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
