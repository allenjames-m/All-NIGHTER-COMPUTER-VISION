"""Run from the repo root:  python eval/test_validate.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from validate import validate

def kinds(issues, level):
    return [i["what"] for i in issues if i["level"] == level]

# --- 1. clean plan: no problems -------------------------------------------------
clean = {"walls": [
    {"id": "a", "p1": [0, 0], "p2": [5, 0]}, {"id": "b", "p1": [5, 0], "p2": [5, 4]},
    {"id": "c", "p1": [5, 4], "p2": [0, 4]}, {"id": "d", "p1": [0, 4], "p2": [0, 0]}],
    "doors": [{"wall": "a", "pos": 1.5, "width": 0.9}],
    "windows": [{"wall": "c", "pos": 2.5, "width": 1.2}],
    "rooms": [{"name": "A", "polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]}]}
_, issues = validate(clean)
assert [i for i in issues if i["level"] != "fixed"] == [] and issues == [], issues

# --- 2. broken plan: gap, zero wall, door off the wall, window inside, no rooms ---
broken = {"walls": [
    {"id": "a", "p1": [0, 0], "p2": [6, 0]},
    {"id": "b", "p1": [6, 0], "p2": [6, 4]},
    {"id": "c", "p1": [6, 4], "p2": [0, 4]},
    {"id": "d", "p1": [0, 3.8], "p2": [0, 0]},          # 20 cm gap to wall c
    {"id": "z", "p1": [2, 2], "p2": [2, 2]},            # zero length
    {"id": "m", "p1": [3, 0], "p2": [3, 4]}],           # interior partition
    "doors": [{"wall": "a", "pos": 5.9, "width": 0.9}],  # sticks out of the wall
    "windows": [{"wall": "m", "pos": 2.0, "width": 1.0}],  # on an interior wall
    "rooms": []}
fixed, issues = validate(broken)
fx, wr = kinds(issues, "fixed"), kinds(issues, "warning")
assert any("zero-length" in s for s in fx), issues
assert any("gap" in s for s in fx), issues
assert any("moved to lie fully" in s for s in fx), issues
assert any("room(s) derived" in s for s in fx), issues
assert any("interior wall" in s for s in wr), issues
d = fixed["doors"][0]
assert abs(d["pos"] - 5.55) < 1e-6 and d["source"] == "inferred", d
assert next(w for w in fixed["walls"] if w["id"] == "d")["source"] == "inferred"
assert len(fixed["rooms"]) == 2, fixed["rooms"]          # partition splits the flat in two

# --- 3. open outline: error ------------------------------------------------------
_, issues = validate({"walls": [{"id": "a", "p1": [0, 0], "p2": [5, 0]}, {"id": "b", "p1": [5, 0], "p2": [5, 4]}]})
assert kinds(issues, "error"), issues

print("Validation tests passed")
for i in validate(broken)[1]:
    print(f"  [{i['level']:7s}] {i['what']}")
