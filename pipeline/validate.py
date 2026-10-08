"""Geometric validation loop for a plan JSON.

Checks (and fixes what can be fixed safely):
  1. zero-length walls                  -> removed
  2. dangling wall ends (gaps in walls) -> extended along their own direction to meet the next wall
  3. closed rooms                       -> rooms are re-derived from the walls; if the
                                           plan has no rooms, they are added (source: assumed)
  4. doors / windows                    -> must sit fully on their wall (position clamped),
                                           must not overlap each other, must fit
  5. windows                            -> should be on exterior walls (warning otherwise)
  6. doors                              -> should connect two rooms or lead outside (warning otherwise)
Anything it changed is marked source = "inferred" so the viewer shows it in yellow.

    fixed_plan, issues = validate(plan)
    issues = [{"level": "fixed"|"warning"|"error", "what": "...", "id": "w3"}, ...]
Usage:  python pipeline/validate.py plan.json [out.json]
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def _L(w):
    return float(np.hypot(w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]))


def _dist_pt_seg(pt, a, b):
    """distance from pt to segment ab, closest point on ab"""
    pt, a, b = np.array(pt, float), np.array(a, float), np.array(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-12 else float(np.clip((pt - a) @ ab / L2, 0, 1))
    c = a + t * ab
    return float(np.hypot(*(pt - c))), c


def _faces(plan):
    lines = unary_union([LineString([w["p1"], w["p2"]]) for w in plan["walls"] if _L(w) > 1e-6])
    # snap corners to a 5 mm grid so wall ends that meet "almost exactly" really connect
    lines = shapely.set_precision(lines, 0.005)
    return [f for f in polygonize(lines) if f.area > 0.5]


def validate(plan, snap_tol=0.4, node_tol=0.05):
    plan = copy.deepcopy(plan)
    issues = []

    def log(level, what, id_=None):
        issues.append({"level": level, "what": what, "id": id_})

    # 1. zero-length walls
    keep = []
    for w in plan["walls"]:
        if _L(w) < 0.02:
            log("fixed", "zero-length wall removed", w["id"])
        else:
            keep.append(w)
    plan["walls"] = keep

    # 2. dangling ends: an end that touches no other wall end
    def touching(pt, skip):
        for o in plan["walls"]:
            if o is skip:
                continue
            d, _ = _dist_pt_seg(pt, o["p1"], o["p2"])
            if d < node_tol:
                return True
        return False

    def extend_to_wall(w, key):
        """Slide the end `key` of wall w along its own direction (extend or trim) until it
        meets another wall. Keeps the wall straight. Returns (new_point, wall_id) or None."""
        other = "p2" if key == "p1" else "p1"
        end, start = np.array(w[key], float), np.array(w[other], float)
        u = (end - start) / max(np.hypot(*(end - start)), 1e-9)
        best = None
        for o in plan["walls"]:
            if o is w:
                continue
            a, b = np.array(o["p1"], float), np.array(o["p2"], float)
            ab = b - a
            M = np.array([[u[0], -ab[0]], [u[1], -ab[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue                                  # parallel walls never meet
            s_, t = np.linalg.solve(M, a - end)           # end + s*u = a + t*ab
            slack = snap_tol / max(np.hypot(*ab), 1e-9)   # allow reaching just past the other wall's end
            if abs(s_) <= snap_tol and -slack <= t <= 1 + slack:
                if best is None or abs(s_) < abs(best[0]):
                    best = (s_, end + s_ * u, o["id"])
        return best

    for w in plan["walls"]:
        for key in ("p1", "p2"):
            if touching(w[key], w):
                continue
            hit = extend_to_wall(w, key)
            if hit and abs(hit[0]) > 1e-3:
                w[key] = [float(hit[1][0]), float(hit[1][1])]
                w["source"] = "inferred"
                log("fixed", f"gap of {abs(hit[0])*100:.0f} cm closed (end {key} extended to wall {hit[2]})", w["id"])
    for w in plan["walls"]:                               # whatever is still loose gets a warning
        for key in ("p1", "p2"):
            if not touching(w[key], w):
                log("warning", f"wall end {key} is not connected to any wall", w["id"])

    # 3. closed rooms
    faces = _faces(plan)
    if not faces:
        log("error", "walls enclose no room (outline not closed)")
    elif not plan.get("rooms"):
        plan["rooms"] = [{"name": f"room_{i+1}", "source": "assumed",
                          "polygon": [list(map(float, p)) for p in f.exterior.coords[:-1]]}
                         for i, f in enumerate(faces)]
        log("fixed", f"{len(faces)} room(s) derived from the walls")
    else:
        for r in plan["rooms"]:
            rp = Polygon(r["polygon"])
            rp = rp if rp.is_valid else rp.buffer(0)
            ok = any(rp.union(f).area > 0 and rp.intersection(f).area / rp.union(f).area > 0.5 for f in faces)
            if not ok:
                log("warning", f"room '{r.get('name', '?')}' is not enclosed by walls", r.get("name"))

    # wall -> number of rooms beside it (0 = outside, 1 = exterior wall, 2 = shared)
    def n_sides(w):
        mid = [(w["p1"][0] + w["p2"][0]) / 2, (w["p1"][1] + w["p2"][1]) / 2]
        return sum(1 for f in faces if f.exterior.distance(Point(mid)) < 0.1)

    walls = {w["id"]: w for w in plan["walls"]}

    # 4-6. openings
    for kind in ("doors", "windows"):
        taken = {}
        for o in plan.get(kind, []):
            w = walls.get(o["wall"])
            label = f"{kind[:-1]} on {o['wall']}"
            if w is None:
                log("error", f"{label}: wall does not exist")
                continue
            L, half = _L(w), o.get("width", 0.9) / 2
            if half * 2 > L:
                log("error", f"{label}: wider ({half*2:.2f} m) than its wall ({L:.2f} m)", o["wall"])
                continue
            if o["pos"] < half or o["pos"] > L - half:
                o["pos"] = float(np.clip(o["pos"], half, L - half))
                o["source"] = "inferred"
                log("fixed", f"{label}: moved to lie fully on the wall", o["wall"])
            for a, b in taken.get(o["wall"], []):
                if o["pos"] - half < b and o["pos"] + half > a:
                    log("warning", f"{label}: overlaps another opening", o["wall"])
            taken.setdefault(o["wall"], []).append((o["pos"] - half, o["pos"] + half))
            sides = n_sides(w) if faces else None
            if sides is not None and kind == "windows" and sides >= 2:
                log("warning", f"{label}: window on an interior wall", o["wall"])
            if sides is not None and kind == "doors" and sides == 0:
                log("warning", f"{label}: door on a wall that borders no room", o["wall"])
    return plan, issues


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    fixed, issues = validate(plan)
    for i in issues:
        print(f"[{i['level']:7s}] {i['what']}" + (f"  ({i['id']})" if i["id"] else ""))
    if not issues:
        print("No problems found.")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(fixed, indent=2), encoding="utf-8")
        print(f"Wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
