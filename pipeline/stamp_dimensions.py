"""Draw TRUE dimension labels onto a plan image (the dataset's plans have none).

Why: the dimension solver needs written measurements. CubiCasa images only show room names,
so we write the real wall lengths (from the ground-truth JSON) onto copies of the images.
Say so in the write-up: "synthetic dimension annotations on real plans".

    python pipeline/stamp_dimensions.py data/gt/plan1.json data/plans/plan1.png data/stamped
    python pipeline/stamp_dimensions.py --all data            # every plan in data/gt

Writes   <out>/<name>.png               plan + dimension lines and labels
         <out>/<name>_ocr_truth.json    [{"text","center":[x,y] (metres),"wall":id}]  = what a perfect OCR would return
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPoint


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick_walls(plan, min_len=1.2, max_labels=14):
    """long, straight walls; one label per wall; spread out so labels don't pile up"""
    cands = []
    for w in plan["walls"]:
        a, b = np.array(w["p1"]), np.array(w["p2"])
        L = float(np.hypot(*(b - a)))
        ang = np.degrees(np.arctan2(abs(b[1] - a[1]), abs(b[0] - a[0])))
        if L >= min_len and (ang < 5 or ang > 85):
            cands.append((L, w))
    cands.sort(key=lambda t: -t[0])
    chosen, mids = [], []
    for L, w in cands:
        mid = (np.array(w["p1"]) + np.array(w["p2"])) / 2
        if all(np.hypot(*(mid - m)) > 0.9 for m in mids):
            chosen.append(w)
            mids.append(mid)
        if len(chosen) >= max_labels:
            break
    return chosen


def stamp(plan, image, offset_m=0.55):
    """returns (new PIL image, ocr_truth list)"""
    ppm = plan["meta"]["px_per_metre"]
    W, H = plan["meta"]["svg_size_px"]
    img = image.convert("RGB")
    if img.size != (round(W), round(H)):
        img = img.resize((round(W), round(H)))          # F1_scaled should already match; be safe
    d = ImageDraw.Draw(img)
    font = _font(max(18, int(0.26 * ppm)))
    to_px = lambda p: (p[0] * ppm, H - p[1] * ppm)       # metres -> pixels (y flipped back)
    inside = MultiPoint([pt for r in plan["rooms"] for pt in r["polygon"]]).centroid
    truth = []
    for w in pick_walls(plan):
        a, b = np.array(w["p1"]), np.array(w["p2"])
        L = float(np.hypot(*(b - a)))
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        mid = (a + b) / 2
        if (mid + n - np.array([inside.x, inside.y])) @ n < 0:   # point the offset away from the flat's centre
            n = -n
        a2, b2, m2 = a + n * offset_m, b + n * offset_m, mid + n * offset_m
        pa, pb, pm = to_px(a2), to_px(b2), to_px(m2)
        d.line([pa, pb], fill=(0, 0, 0), width=2)
        for q, t in ((pa, a2), (pb, b2)):                       # end ticks
            tp = to_px(t + n * 0.08), to_px(t - n * 0.08)
            d.line(tp, fill=(0, 0, 0), width=2)
        text = f"{L:.2f} m"
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        # CAD style: the number sits ON the dimension line (white box interrupts the line), kept horizontal
        cx, cy = pm[0], pm[1]
        d.rectangle([cx - tw / 2 - 3, cy - th / 2 - 3, cx + tw / 2 + 3, cy + th / 2 + 3], fill=(255, 255, 255))
        d.text((cx - tw / 2 - box[0], cy - th / 2 - box[1]), text, fill=(0, 0, 0), font=font)
        truth.append({"text": text, "center": [round(cx / ppm, 3), round((H - cy) / ppm, 3)], "wall": w["id"]})
    return img, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", nargs="?")
    ap.add_argument("image", nargs="?")
    ap.add_argument("out", nargs="?", default="data/stamped")
    ap.add_argument("--all", help="data folder containing gt/ and plans/")
    a = ap.parse_args()
    jobs = []
    if a.all:
        root = Path(a.all)
        for g in sorted((root / "gt").glob("*.json")):
            img = root / "plans" / f"{g.stem}.png"
            if img.exists():
                jobs.append((g, img, root / "stamped"))
    elif a.gt and a.image:
        jobs.append((Path(a.gt), Path(a.image), Path(a.out)))
    else:
        ap.error("give: gt.json image.png [out_dir]   or   --all data")
    for g, img, out in jobs:
        out.mkdir(parents=True, exist_ok=True)
        plan = json.loads(g.read_text(encoding="utf-8"))
        im, truth = stamp(plan, Image.open(img))
        im.save(out / f"{g.stem}.png")
        (out / f"{g.stem}_ocr_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
        print(f"{g.stem}: {len(truth)} dimensions drawn")
    print(f"\nDone: {len(jobs)} plan(s)")


if __name__ == "__main__":
    main()
