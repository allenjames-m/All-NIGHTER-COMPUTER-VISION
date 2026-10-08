"""Fake-data test: walls -> GLB. Proves the export-and-view path works.
Run from anywhere:  python pipeline/build_3d.py
Output: viewer/test_room.glb
"""
from pathlib import Path
import numpy as np
import trimesh

WALL_H, WALL_T = 2.7, 0.15
DOOR_H, DOOR_W = 2.1, 0.9

# provenance colours: green = detected, yellow = inferred, red = assumed default
GREEN, YELLOW, RED = [60, 180, 75, 255], [240, 200, 40, 255], [220, 60, 60, 255]


def wall_box(x1, y1, x2, y2, z0, z1, color):
    """Wall piece from (x1,y1) to (x2,y2), from height z0 to z1."""
    length = np.hypot(x2 - x1, y2 - y1)
    h = z1 - z0
    box = trimesh.creation.box(extents=[length, WALL_T, h])
    angle = np.arctan2(y2 - y1, x2 - x1)
    T = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    T[:3, 3] = [(x1 + x2) / 2, (y1 + y2) / 2, z0 + h / 2]
    box.apply_transform(T)
    box.visual.face_colors = color
    return box


def wall_with_door(x1, y1, x2, y2, door_pos, color):
    """Wall split into left piece, right piece and a lintel above the door.
    door_pos = distance (m) from wall start to door centre."""
    L = np.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    a, b = door_pos - DOOR_W / 2, door_pos + DOOR_W / 2
    p = lambda d: (x1 + ux * d, y1 + uy * d)
    return [
        wall_box(*p(0), *p(a), 0, WALL_H, color),
        wall_box(*p(b), *p(L), 0, WALL_H, color),
        wall_box(*p(a), *p(b), DOOR_H, WALL_H, color),
    ]


def main():
    meshes = []
    # 5 x 4 m room, one door in the bottom wall
    meshes += wall_with_door(0, 0, 5, 0, door_pos=1.5, color=GREEN)
    meshes.append(wall_box(5, 0, 5, 4, 0, WALL_H, GREEN))
    meshes.append(wall_box(5, 4, 0, 4, 0, WALL_H, YELLOW))  # pretend: solved from dimensions
    meshes.append(wall_box(0, 4, 0, 0, 0, WALL_H, GREEN))
    floor = trimesh.creation.box(extents=[5, 4, 0.1])
    floor.apply_translation([2.5, 2, -0.05])
    floor.visual.face_colors = RED  # assumed default
    meshes.append(floor)

    scene = trimesh.Scene(meshes)
    # glTF uses Y-up; our model is Z-up, so rotate so the room stands upright
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    out = Path(__file__).resolve().parent.parent / "viewer" / "test_room.glb"
    out.parent.mkdir(exist_ok=True)
    scene.export(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
