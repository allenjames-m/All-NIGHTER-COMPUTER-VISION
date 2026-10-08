"""Run from the repo root:  python eval/test_dims.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from dims import link_to_walls, parse_dimension

def m(text):
    d = parse_dimension(text)
    return None if d is None else round(d["metres"], 3)

# parsing
assert m("5.00 m") == 5.0
assert m("3,50m") == 3.5
assert m("350 cm") == 3.5
assert m("4200") == 4.2                   # bare integer = millimetres (guessed)
assert m("12'6\"") == round(12 * 0.3048 + 6 * 0.0254, 3)
assert m("9'-0\"") == round(9 * 0.3048, 3)
assert m("5.O0 m") == 5.0                 # OCR read letter O for zero
assert m("LIVING") is None and m("") is None
assert parse_dimension("4200")["confident"] is False
assert parse_dimension("5.00 m")["confident"] is True

# linking: plan from data/sample_plan.json style 5 x 4 room
plan = {"walls": [
    {"id": "w1", "p1": [0, 0], "p2": [5, 0]}, {"id": "w2", "p1": [5, 0], "p2": [5, 4]},
    {"id": "w3", "p1": [5, 4], "p2": [0, 4]}, {"id": "w4", "p1": [0, 4], "p2": [0, 0]}]}
items = [
    {"text": "5.00 m", "center": [2.5, -0.6]},    # below wall w1
    {"text": "4000", "center": [5.7, 2.0]},       # beside wall w2 (mm, guessed)
    {"text": "KITCHEN", "center": [2.5, 2.0]},    # not a dimension
    {"text": "9.90 m", "center": [2.5, 4.5]},     # misread: implausible for w3
]
cons, skipped = link_to_walls(items, plan)
got = {c["wall"]: c["length"] for c in cons}
assert got == {"w1": 5.0, "w2": 4.0}, got
assert len(skipped) == 1 and skipped[0]["text"] == "9.90 m", skipped
print("Dimension parsing and linking tests passed")
print("constraints:", cons)
print("skipped:", [(s["text"], s["reason"]) for s in skipped])
