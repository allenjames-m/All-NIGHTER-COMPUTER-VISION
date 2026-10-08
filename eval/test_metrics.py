"""Sanity tests with known answers. Run:  python eval/test_metrics.py"""
import copy
import math
from metrics import evaluate

GT = {
    "walls": [
        {"id": "w1", "p1": [0, 0], "p2": [5, 0]}, {"id": "w2", "p1": [5, 0], "p2": [5, 4]},
        {"id": "w3", "p1": [5, 4], "p2": [0, 4]}, {"id": "w4", "p1": [0, 4], "p2": [0, 0]}],
    "doors": [{"wall": "w1", "pos": 1.5}],
    "windows": [{"wall": "w3", "pos": 2.5}],
    "rooms": [{"name": "A", "polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]}],
}

def close(a, b, tol=0.02):
    return abs(a - b) <= tol

# 1. identical plan -> perfect scores
s = evaluate(GT, GT)
assert s["layout_iou"] == 1.0 and s["dim_error_cm"] == 0.0 and s["door_f1"] == 1.0, s

# 2. room 10% too wide (5.0 -> 5.5): IoU = 20/22, width error 50 cm
P = copy.deepcopy(GT)
P["rooms"][0]["polygon"] = [[0, 0], [5.5, 0], [5.5, 4], [0, 4]]
s = evaluate(GT, P)
assert close(s["layout_iou"], 20 / 22, 0.001), s
# room bbox errors: 50cm, 0cm  -> mean 25 (walls unchanged add 0)
assert s["dim_error_cm"] > 0, s

# 3. door moved 2 m away -> door F1 = 0, window still found
P = copy.deepcopy(GT)
P["doors"][0]["pos"] = 3.5
s = evaluate(GT, P)
assert s["door_f1"] == 0.0 and s["window_f1"] == 1.0, s

# 4. missing everything -> zeros, no crash
s = evaluate(GT, {"walls": [], "rooms": []})
assert s["layout_iou"] == 0.0 and s["door_f1"] == 0.0 and math.isnan(s["dim_error_cm"]), s

print("All metric tests passed")
