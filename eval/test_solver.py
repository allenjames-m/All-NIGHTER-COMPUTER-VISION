"""Synthetic test: noisy detection + OCR dimensions (with one OCR mistake).
Run from the repo root:  python eval/test_solver.py
Shows dimension error before / after each step, using eval/metrics.py."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import evaluate
from solver import snap_manhattan, solve_dimensions

# ground truth: an L-shaped flat, 6 walls
GT = {
    "walls": [
        {"id": "a", "p1": [0, 0], "p2": [6, 0], "source": "detected"},
        {"id": "b", "p1": [6, 0], "p2": [6, 2], "source": "detected"},
        {"id": "c", "p1": [6, 2], "p2": [3, 2], "source": "detected"},
        {"id": "d", "p1": [3, 2], "p2": [3, 5], "source": "detected"},
        {"id": "e", "p1": [3, 5], "p2": [0, 5], "source": "detected"},
        {"id": "f", "p1": [0, 5], "p2": [0, 0], "source": "detected"},
    ],
    "doors": [{"wall": "a", "pos": 1.5, "width": 0.9}],
    "windows": [{"wall": "e", "pos": 1.5, "width": 1.2}],
    "rooms": [{"name": "L", "polygon": [[0, 0], [6, 0], [6, 2], [3, 2], [3, 5], [0, 5]]}],
}


def noisy_copy(plan, sigma, rng):
    """Jitter every shared corner by gaussian noise (as a detector would)."""
    p = copy.deepcopy(plan)
    corners = {}
    def jitter(pt):
        key = tuple(pt)
        if key not in corners:
            corners[key] = [pt[0] + rng.normal(0, sigma), pt[1] + rng.normal(0, sigma)]
        return corners[key]
    for w in p["walls"]:
        w["p1"], w["p2"] = jitter(w["p1"]), jitter(w["p2"])
    for r in p["rooms"]:
        r["polygon"] = [jitter(pt) for pt in r["polygon"]]
    return p


def wall_len(w):
    return float(np.hypot(w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]))


rng = np.random.default_rng(0)
rows = {"detected only": [], "+ Manhattan snap": [], "+ dimension solver": []}
for trial in range(30):
    pred = noisy_copy(GT, sigma=0.12, rng=rng)
    # OCR reads 4 of the 6 dimensions; one reading is wrong (a misread "3.0" -> "30")
    cons = [{"wall": w["id"], "length": wall_len(w)} for w in GT["walls"]
            if w["id"] in ("a", "b", "d", "e")]
    cons[1]["length"] = 20.0
    snapped = snap_manhattan(pred)
    solved, rep = solve_dimensions(snapped, cons)
    for name, plan in (("detected only", pred), ("+ Manhattan snap", snapped),
                       ("+ dimension solver", solved)):
        rows[name].append(evaluate(GT, plan))

print(f"{'variant':22s} {'layout IoU':>10s} {'dim err (cm)':>13s}")
for name, res in rows.items():
    iou = np.mean([r["layout_iou"] for r in res])
    dim = np.nanmean([r["dim_error_cm"] for r in res])
    print(f"{name:22s} {iou:10.3f} {dim:13.1f}")
print("\nlast trial report:", json.dumps({k: rep[k] for k in ("used", "moved_walls")}),
      "| rejected:", [(r["wall"], r["reason"]) for r in rep["rejected"]])

assert np.nanmean([r["dim_error_cm"] for r in rows["+ dimension solver"]]) < \
       np.nanmean([r["dim_error_cm"] for r in rows["detected only"]])
print("\nSolver test passed: dimension error went down.")
