"""One command: detected plan (+ OCR text) -> checked, dimension-corrected, coloured 3D model.

    python pipeline/run_pipeline.py data/demo_detected.json data/demo_ocr.json --gt data/demo_gt.json

Stages (each saved to out/<plan name>/ so you can show them or score them):
    0_detected.json    what the detector gave us
    1_snapped.json     + Manhattan snapping
    2_solved.json      + dimension solver (uses OCR text)
    3_validated.json   + validation loop            <- final plan
Final 3D model: viewer/model.glb   (green = detected, yellow = inferred, red = assumed)

OCR file: [{"text": "6.00 m", "center": [x, y]}, ...]   center in plan coordinates (metres)
--gt:     ground-truth plan JSON; if given, every stage is scored (the ablation table)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))

from build_3d import build_scene          # noqa: E402
from dims import link_to_walls            # noqa: E402
from solver import snap_manhattan, solve_dimensions  # noqa: E402
from validate import validate             # noqa: E402


def save(plan, path):
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def run(detected, ocr_items=None, gt=None, out_dir=None, glb_path=None, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    stages = {"0_detected": detected}

    stages["1_snapped"] = snap_manhattan(detected)

    constraints, skipped = [], []
    if ocr_items:
        constraints, skipped = link_to_walls(ocr_items, stages["1_snapped"])
        log(f"OCR: {len(constraints)} dimensions linked to walls, {len(skipped)} ignored")
        for s in skipped:
            log(f"   ignored '{s['text']}': {s['reason']}")
    solved, report = solve_dimensions(stages["1_snapped"], constraints)
    for r in report["rejected"]:
        log(f"   rejected reading for wall {r['wall']}: {r['reason']}")
    stages["2_solved"] = solved

    validated, issues = validate(solved)
    stages["3_validated"] = validated
    for i in issues:
        log(f"   [{i['level']}] {i['what']}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, plan in stages.items():
            save(plan, out_dir / f"{name}.json")
    if glb_path:
        glb_path.parent.mkdir(exist_ok=True)
        build_scene(validated).export(glb_path)
        log(f"3D model written: {glb_path}")

    scores = None
    if gt:
        from metrics import evaluate
        scores = {name: evaluate(gt, plan) for name, plan in stages.items()}
        log(f"\n{'stage':14s} {'layout IoU':>10s} {'dim err cm':>11s} {'door F1':>8s} {'window F1':>10s}")
        for name, s in scores.items():
            log(f"{name:14s} {s['layout_iou']:10.3f} {s['dim_error_cm']:11.1f} {s['door_f1']:8.2f} {s['window_f1']:10.2f}")
    return stages, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detected")
    ap.add_argument("ocr", nargs="?")
    ap.add_argument("--gt")
    ap.add_argument("--out", default=None, help="folder for stage files (default out/<plan name>)")
    a = ap.parse_args()

    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    detected = load(a.detected)
    ocr = load(a.ocr) if a.ocr else None
    gt = load(a.gt) if a.gt else None
    out_dir = Path(a.out) if a.out else ROOT / "out" / Path(a.detected).stem
    run(detected, ocr, gt, out_dir, ROOT / "viewer" / "model.glb")


if __name__ == "__main__":
    main()
