"""CubiCasa5K model.svg  ->  ground-truth plan JSON (metres, same format as the rest of the pipeline).

One plan:   python pipeline/svg_to_plan.py path/to/model.svg out.json
Many plans: python pipeline/svg_to_plan.py --batch C:\\...\\cubicasa5k --split test.txt --n 30 --out data

How it works
  * Scale: every room in the SVG carries a hidden label like "2.24 m x 1.76 m". Comparing that with the
    room's size in SVG units gives pixels-per-metre (about 100 for this dataset). Median over all rooms.
  * Walls : each <g class="Wall ..."> holds a quad; its centre line becomes p1/p2, and its thickness is kept.
  * Doors / windows: groups nested inside a wall (or nearest wall if not nested); position = distance of
    the opening's centre from p1 along the wall.
  * Rooms : <g class="Space ..."> polygons (outdoor areas are skipped).
  * Y is flipped so the plan is not mirrored when viewed from above.
Images: use F1_scaled.png, NOT F1_original.png (only the scaled one lines up with the SVG coordinates).
"""
import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

NS = "{http://www.w3.org/2000/svg}"
DIM_RE = re.compile(r"([\d.]+)\s*m\s*x\s*([\d.]+)\s*m")


def _cls(el):
    return el.get("class") or ""


def _points(poly_el):
    pts = []
    for p in poly_el.get("points", "").replace(",", " ").split():
        pts.append(float(p))
    return list(zip(pts[0::2], pts[1::2]))


def _first_polygon(g):
    """polygon that belongs to g itself (direct child), not to nested groups"""
    for ch in g:
        if ch.tag == NS + "polygon":
            pts = _points(ch)
            if len(pts) >= 3:
                return pts
    return None


def _clean_polygon(pts):
    P = Polygon(pts).buffer(0)
    if P.is_empty:
        return None
    if P.geom_type != "Polygon":
        P = max(P.geoms, key=lambda x: x.area)
    return P


def estimate_scale(root):
    """pixels per metre, from the hidden 'W m x H m' room labels"""
    ratios = []
    for g in root.iter(NS + "g"):
        c = _cls(g)
        if not c.startswith("Space ") or "Label" in c:
            continue
        pts = _first_polygon(g)
        P = _clean_polygon(pts) if pts else None
        if P is None:
            continue
        x0, y0, x1, y1 = P.bounds
        for t in g.iter(NS + "text"):
            m = DIM_RE.fullmatch((t.text or "").strip())
            if m:
                w, h = float(m.group(1)), float(m.group(2))
                if w > 0.3 and h > 0.3:
                    ratios += [(x1 - x0) / w, (y1 - y0) / h]
    if len(ratios) < 4:
        return None, None
    a = np.array(ratios)
    med = float(np.median(a))
    close = float(np.mean(np.abs(a - med) / med < 0.05))   # share of readings agreeing within 5 %
    return med, close


def _centerline(pts):
    """centre line of a wall quad: axis = its longest edge. Returns (p1, p2, thickness) in px."""
    P = np.array(pts, float)
    edges = [(P[(i + 1) % len(P)] - P[i]) for i in range(len(P))]
    e = max(edges, key=lambda v: np.hypot(*v))
    L = np.hypot(*e)
    if L < 1e-6:
        return None
    ax = e / L
    nm = np.array([-ax[1], ax[0]])
    a, n = P @ ax, P @ nm
    nc = (n.min() + n.max()) / 2
    p1 = ax * a.min() + nm * nc
    p2 = ax * a.max() + nm * nc
    return p1, p2, float(n.max() - n.min())


def convert(svg_path):
    root = ET.parse(svg_path).getroot()
    H = float(root.get("viewBox", "0 0 0 0").split()[3])
    scale, agree = estimate_scale(root)
    if not scale:
        raise ValueError("no room dimension labels, cannot determine scale")
    to_m = lambda x, y: [round(x / scale, 3), round((H - y) / scale, 3)]

    parent = {c: p for p in root.iter() for c in p}
    walls, wall_geo = [], []
    for g in root.iter(NS + "g"):
        c = _cls(g)
        if c == "Wall" or c == "Wall External":
            pts = _first_polygon(g)
            cl = _centerline(pts) if pts else None
            if cl is None:
                continue
            p1, p2, th = cl
            if np.hypot(*(p2 - p1)) / scale < 0.2:     # tiny stubs / pillars
                continue
            wid = f"w{len(walls) + 1}"
            walls.append({"id": wid, "p1": to_m(*p1), "p2": to_m(*p2),
                          "thickness": round(th / scale, 3), "source": "detected",
                          "external": c == "Wall External"})
            wall_geo.append((g, p1, p2))

    def nearest_wall(centre):
        best = None
        for k, (_, p1, p2) in enumerate(wall_geo):
            ab = p2 - p1
            t = np.clip((centre - p1) @ ab / max(ab @ ab, 1e-9), 0, 1)
            d = np.hypot(*(centre - (p1 + t * ab)))
            if best is None or d < best[0]:
                best = (d, k)
        return best[1] if best else None

    doors, windows = [], []
    for g in root.iter(NS + "g"):
        c = _cls(g)
        kind = "doors" if c.startswith("Door") else "windows" if c.startswith("Window") else None
        if kind is None or g.get("id") not in ("Door", "Window"):
            continue
        pts = _first_polygon(g)
        if not pts:
            continue
        P = np.array(pts, float)
        centre = P.mean(axis=0)
        k = None                                           # nested in a wall? use that wall
        node = parent.get(g)
        while node is not None and k is None:
            for j, (wg, _, _) in enumerate(wall_geo):
                if wg is node:
                    k = j
            node = parent.get(node)
        if k is None:
            k = nearest_wall(centre)
        if k is None:
            continue
        _, p1, p2 = wall_geo[k]
        ab = p2 - p1
        L = np.hypot(*ab)
        ax = ab / L
        proj = P @ ax
        width = float(proj.max() - proj.min()) / scale
        pos = float(((proj.max() + proj.min()) / 2) - p1 @ ax) / scale
        item = {"wall": walls[k]["id"], "pos": round(pos, 3), "width": round(width, 3), "source": "detected"}
        (doors if kind == "doors" else windows).append(item)

    rooms = []
    for g in root.iter(NS + "g"):
        c = _cls(g)
        if not c.startswith("Space ") or "Label" in c or "Outdoor" in c:
            continue
        pts = _first_polygon(g)
        P = _clean_polygon(pts) if pts else None
        if P is None or P.area / scale ** 2 < 0.5:
            continue
        name = next((t.text for t in g.iter(NS + "text") if t.text and not DIM_RE.search(t.text)
                     and "'" not in t.text), c.split()[1])
        coords = list(P.simplify(0.5).exterior.coords)[:-1]
        rooms.append({"name": name, "type": c[6:], "polygon": [to_m(x, y) for x, y in coords]})

    return {"wall_height": 2.7, "walls": walls, "doors": doors, "windows": windows, "rooms": rooms,
            "meta": {"source_svg": str(svg_path), "px_per_metre": round(scale, 2),
                     "svg_size_px": [round(float(root.get("viewBox").split()[2]), 1), round(H, 1)],
                     "scale_agreement": round(agree, 2),
                     "note": "ground truth derived from CubiCasa annotations; metres; y flipped"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svg", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--batch", help="dataset root (folder containing high_quality_architectural etc.)")
    ap.add_argument("--split", default="test.txt", help="list file inside the dataset root")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", default="high_quality_architectural", help="only plans from this folder ('' = all)")
    ap.add_argument("--out", dest="out_dir", default="data")
    a = ap.parse_args()

    if not a.batch:
        if not a.svg or not a.out:
            ap.error("give: model.svg out.json   (or use --batch)")
        plan = convert(a.svg)
        Path(a.out).write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"{len(plan['walls'])} walls, {len(plan['doors'])} doors, {len(plan['windows'])} windows, "
              f"{len(plan['rooms'])} rooms, scale {plan['meta']['px_per_metre']} px/m -> {a.out}")
        return

    root = Path(a.batch)
    lines = [l.strip() for l in (root / a.split).read_text().splitlines() if l.strip()]
    out = Path(a.out_dir)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    done = 0
    for line in lines:
        rel = line.strip("/").replace("\\", "/")
        if a.only and not rel.startswith(a.only):
            continue
        folder = root / rel
        svg = folder / "model.svg"
        if not svg.exists():
            continue
        try:
            plan = convert(svg)
        except Exception as e:
            print(f"skip {rel}: {e}")
            continue
        if len(plan["walls"]) < 4 or len(plan["rooms"]) < 1 or plan["meta"]["scale_agreement"] < 0.8:
            print(f"skip {rel}: too few walls/rooms or inconsistent scale")
            continue
        pid = rel.split("/")[-1]
        (out / "gt" / f"plan{pid}.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        img = folder / "F1_scaled.png"
        if img.exists():
            shutil.copy(img, out / "plans" / f"plan{pid}.png")
        print(f"plan{pid}: {len(plan['walls'])} walls, {len(plan['doors'])} doors, "
              f"{len(plan['windows'])} windows, {len(plan['rooms'])} rooms")
        done += 1
        if done >= a.n:
            break
    print(f"\nConverted {done} plans into {out}/gt and {out}/plans")


if __name__ == "__main__":
    main()
