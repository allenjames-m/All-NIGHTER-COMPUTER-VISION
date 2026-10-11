"""The ablation table: each stage of the pipeline, scored against ground truth on all test plans.

    python eval/ablation.py data                      # simulated detector (controlled experiment)
    python eval/ablation.py data --pred-dir preds     # YOUR detector: preds/planNNN.json per plan

Stages:  0 detector output | 1 + scale from dimension text | 2 + snapping | 3 + dimension solver | 4 + validation
Needs:   data/gt/planNNN.json and data/stamped/planNNN_ocr_truth.json  (from svg_to_plan.py / stamp_dimensions.py)

The OCR text comes from the *_ocr_truth.json files = a PERFECT reader. Replace them with real OCR output
(same format) to measure the effect of OCR mistakes.
--pred-dir files must use the same frame as the ground truth: run your parser on data/stamped/planNNN.png
(100 px per metre), y flipped (y_m = (image_height - y_px) / 100). Rooms are required; walls/doors optional.

Simulated detector (--detector sim, default): ground-truth geometry where every straight wall line is shifted
by ~8 cm (like a detector that finds lines slightly off), corners jitter ~1 cm, door positions shift ~10 cm, and
the whole plan is scaled by a random +-6 % (as if the pixels-per-metre guess were wrong). Say so in the write-up:
it tests whether the dimension steps help, it is NOT a parser comparison.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE))
from dims import link_to_walls                                              # noqa: E402
from metrics import evaluate                                                # noqa: E402
from solver import (_cluster, _refs, calibrate_scale, snap_lines, solve_lines)  # noqa: E402
from validate import validate                                               # noqa: E402

STAGES = ["0 detector", "1 + scale from text", "2 + snapping", "3 + dimension solver", "4 + validation"]


def simulate_detector(gt, items, rng, sigma_line=0.08, jitter=0.01, door_sigma=0.10, scale_err=0.06):
    p = copy.deepcopy(gt)
    refs = _refs(p)
    for axis in (0, 1):
        vals = np.array([c[k][axis] for c, k in refs])
        lab = _cluster(vals, 0.03)
        off = rng.normal(0, sigma_line, lab.max() + 1)
        for i, (c, k) in enumerate(refs):
            c[k] = list(c[k])
            c[k][axis] += off[lab[i]] + rng.normal(0, jitter)
    for key in ("doors", "windows"):
        for o in p.get(key, []):
            o["pos"] += rng.normal(0, door_sigma)
    f = 1 + rng.uniform(-scale_err, scale_err)
    for w in p["walls"]:
        w["p1"] = [w["p1"][0] * f, w["p1"][1] * f]
        w["p2"] = [w["p2"][0] * f, w["p2"][1] * f]
    for r in p["rooms"]:
        r["polygon"] = [[x * f, y * f] for x, y in r["polygon"]]
    for key in ("doors", "windows"):
        for o in p.get(key, []):
            o["pos"] *= f
            o["width"] = o.get("width", 0.9) * f
    p["doors_windows_note"] = "simulated"
    items2 = [{**it, "center": [it["center"][0] * f, it["center"][1] * f]} for it in items]
    return p, items2, f


def run_stages(pred, items, gt):
    out = {}
    cons, _ = link_to_walls(items, pred)
    out[STAGES[0]] = pred
    p1, f, items1 = calibrate_scale(pred, cons, items)
    out[STAGES[1]] = p1
    p2 = snap_lines(p1)
    out[STAGES[2]] = p2
    cons2, _ = link_to_walls(items1, p2)
    p3, rep = solve_lines(p2, cons2)
    out[STAGES[3]] = p3
    p4, issues = validate(p3)
    out[STAGES[4]] = p4
    return out, f, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--pred-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-line", type=float, default=0.08)
    ap.add_argument("--scale-err", type=float, default=0.06)
    ap.add_argument("--jitter", type=float, default=0.01, help="per-corner noise (m) of the simulated detector")
    ap.add_argument("--repeats", type=int, default=5, help="simulated noise draws per plan")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data = Path(a.data)
    rng = np.random.default_rng(a.seed)
    rows = {s: [] for s in STAGES}
    n_plans = 0
    for g in sorted((data / "gt").glob("*.json")):
        name = g.stem
        ocr = data / "stamped" / f"{name}_ocr_truth.json"
        if not ocr.exists():
            continue
        gt = json.loads(g.read_text(encoding="utf-8"))
        items = json.loads(ocr.read_text(encoding="utf-8"))
        if a.pred_dir:
            pf = Path(a.pred_dir) / f"{name}.json"
            if not pf.exists():
                print(f"no prediction for {name}, skipped")
                continue
            jobs = [(json.loads(pf.read_text(encoding="utf-8")), items)]
        else:
            jobs = []
            for _ in range(a.repeats):
                pred, it, _f = simulate_detector(gt, items, rng, a.sigma_line, jitter=a.jitter, scale_err=a.scale_err)
                jobs.append((pred, it))
        n_plans += 1
        acc = {s: [] for s in STAGES}
        for pred, it in jobs:
            stages, f, rep = run_stages(pred, it, gt)
            for s in STAGES:
                acc[s].append(evaluate(gt, stages[s]))
        for s in STAGES:
            rows[s].append({k: float(np.nanmean([e[k] for e in acc[s]])) for k in acc[s][0]})
    if n_plans == 0:
        print("no plans found (need data/gt and data/stamped/*_ocr_truth.json)")
        return
    cols = [("layout_iou", "Layout IoU", "{:.3f}"), ("room_iou_mean", "Room IoU", "{:.3f}"),
            ("dim_error_cm", "Dim error (cm)", "{:.1f}"), ("door_f1", "Door F1", "{:.2f}"),
            ("window_f1", "Window F1", "{:.2f}")]
    lines = ["| Stage | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for s in STAGES:
        vals = [np.nanmean([r[k] for r in rows[s]]) for k, _, _ in cols]
        lines.append(f"| {s} | " + " | ".join(fmt.format(v) for v, (_, _, fmt) in zip(vals, cols)) + " |")
    mode = "predictions from " + a.pred_dir if a.pred_dir else f"SIMULATED detector ({a.repeats} noise draws/plan, seed {a.seed})"
    table = "\n".join(lines)
    print(f"\n{n_plans} plans, {mode}\n\n{table}\n")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "ablation.md").write_text(f"{n_plans} plans, {mode}\n\n{table}\n", encoding="utf-8")
    print(f"saved {out / 'ablation.md'}")


if __name__ == "__main__":
    main()
