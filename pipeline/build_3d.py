"""Plan JSON -> coloured 3D GLB.

Usage (from the repo root):
    python pipeline/build_3d.py                      # uses data/sample_plan.json
    python pipeline/build_3d.py data/my_plan.json    # any plan JSON
Output: viewer/model.glb   (open viewer/index.html via a local server)

JSON format (metres everywhere):
  walls:   {id, p1:[x,y], p2:[x,y], source}
  doors:   {wall, pos, width, source}                 pos = metres from p1 to door centre
  windows: {wall, pos, width, sill, height, source}
  rooms:   {name, polygon:[[x,y],...]}
  source:  detected (green) | inferred (yellow) | assumed (red)
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent

WALL_T = 0.15
DEFAULT_WALL_H = 2.7
DEFAULT_DOOR_H = 2.1
DEFAULT_DOOR_W = 0.9
DEFAULT_WIN_W = 1.2
DEFAULT_SILL = 0.9
DEFAULT_WIN_H = 1.2

COLORS = {
    "detected": [60, 180, 75, 255],    # green
    "inferred": [240, 200, 40, 255],   # yellow
    "assumed": [220, 60, 60, 255],     # red
}
FLOOR_COLOR = COLORS["assumed"]  # floor is not read from the plan, so it is an assumption


def color_for(source):
    return COLORS.get(source, COLORS["assumed"])


def wall_box(x1, y1, x2, y2, z0, z1, color):
    """Box along the segment (x1,y1)->(x2,y2) from height z0 to z1."""
    length = float(np.hypot(x2 - x1, y2 - y1))
    h = z1 - z0
    if length < 1e-6 or h < 1e-6:
        return None
    box = trimesh.creation.box(extents=[length, WALL_T, h])
    angle = np.arctan2(y2 - y1, x2 - x1)
    T = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    T[:3, 3] = [(x1 + x2) / 2, (y1 + y2) / 2, z0 + h / 2]
    box.apply_transform(T)
    box.visual.face_colors = color
    return box


def build_wall(wall, openings, wall_h):
    """Wall split around its doors/windows. openings = list of dicts with
    kind ('door'|'window'), pos, width, sill, height."""
    x1, y1 = wall["p1"]
    x2, y2 = wall["p2"]
    L = float(np.hypot(x2 - x1, y2 - y1))
    if L < 1e-6:
        return []
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    p = lambda d: (x1 + ux * d, y1 + uy * d)
    color = color_for(wall.get("source"))

    # clamp each opening inside the wall, sort along the wall
    spans = []
    for o in openings:
        a = max(0.0, o["pos"] - o["width"] / 2)
        b = min(L, o["pos"] + o["width"] / 2)
        if b - a > 1e-3:
            spans.append((a, b, o))
    spans.sort(key=lambda s: s[0])

    pieces, cursor = [], 0.0
    for a, b, o in spans:
        a = max(a, cursor)  # ignore overlap with the previous opening
        if a - cursor > 1e-3:
            pieces.append(wall_box(*p(cursor), *p(a), 0, wall_h, color))
        if o["kind"] == "door":
            pieces.append(wall_box(*p(a), *p(b), min(o["height"], wall_h), wall_h, color))
        else:  # window: wall below the sill and above the window head
            head = o["sill"] + o["height"]
            pieces.append(wall_box(*p(a), *p(b), 0, o["sill"], color))
            pieces.append(wall_box(*p(a), *p(b), min(head, wall_h), wall_h, color))
        cursor = max(cursor, b)
    if L - cursor > 1e-3:
        pieces.append(wall_box(*p(cursor), *p(L), 0, wall_h, color))
    return [m for m in pieces if m is not None]


def build_floor(polygon_xy, thickness=0.1):
    """Floor slab for a room polygon (works for non-rectangular rooms)."""
    poly = Polygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    try:
        import shapely
        tris = list(shapely.constrained_delaunay_triangles(poly).geoms)
    except Exception:
        # fallback: bounding box of the room
        minx, miny, maxx, maxy = poly.bounds
        box = trimesh.creation.box(extents=[maxx - minx, maxy - miny, thickness])
        box.apply_translation([(minx + maxx) / 2, (miny + maxy) / 2, -thickness / 2])
        box.visual.face_colors = FLOOR_COLOR
        return box
    meshes = []
    for t in tris:
        xy = np.array(t.exterior.coords)[:3]
        verts = np.array([[x, y, 0.0] for x, y in xy])
        m = trimesh.Trimesh(vertices=verts, faces=[[0, 1, 2]], process=False)
        if m.face_normals[0][2] < 0:  # make the top face point up
            m.invert()
        meshes.append(m)
    floor = trimesh.util.concatenate(meshes)
    floor.visual.face_colors = FLOOR_COLOR
    return floor


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_scene(plan):
    wall_h = plan.get("wall_height", DEFAULT_WALL_H)
    walls = {w["id"]: w for w in plan.get("walls", [])}

    # group openings by wall id
    by_wall = {wid: [] for wid in walls}
    for d in plan.get("doors", []):
        if d["wall"] in by_wall:
            by_wall[d["wall"]].append({
                "kind": "door", "pos": d["pos"],
                "width": d.get("width", DEFAULT_DOOR_W),
                "sill": 0.0, "height": d.get("height", DEFAULT_DOOR_H)})
        else:
            print(f"warning: door refers to unknown wall '{d['wall']}', skipped")
    for w in plan.get("windows", []):
        if w["wall"] in by_wall:
            by_wall[w["wall"]].append({
                "kind": "window", "pos": w["pos"],
                "width": w.get("width", DEFAULT_WIN_W),
                "sill": w.get("sill", DEFAULT_SILL),
                "height": w.get("height", DEFAULT_WIN_H)})
        else:
            print(f"warning: window refers to unknown wall '{w['wall']}', skipped")

    meshes = []
    for wid, wall in walls.items():
        meshes += build_wall(wall, by_wall[wid], wall_h)
    for room in plan.get("rooms", []):
        meshes.append(build_floor(room["polygon"]))

    scene = trimesh.Scene(meshes)
    # glTF is Y-up; our plan is Z-up, so rotate so the room stands upright
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return scene


def main():
    plan_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_plan.json"
    plan = load_plan(plan_path)
    scene = build_scene(plan)
    out = ROOT / "viewer" / "model.glb"
    out.parent.mkdir(exist_ok=True)
    scene.export(out)
    print(f"Built {plan_path.name}: {len(plan.get('walls', []))} walls, "
          f"{len(plan.get('doors', []))} doors, {len(plan.get('windows', []))} windows")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
