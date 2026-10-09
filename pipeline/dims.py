"""Turn OCR text into solver constraints.

1) parse_dimension("12'6\"") -> {"metres": 3.81, "unit": "ft-in", "confident": True}
2) link_to_walls(ocr_items, plan) -> [{"wall": "w1", "length": 5.0, "text": "5.00 m"}, ...]

OCR items:  [{"text": "5.00 m", "center": [x, y]}]   center in PLAN coordinates (metres).
(Convert OCR pixel boxes to metres first, using your scale calibration.)
Feed the result straight into solver.solve_dimensions(plan, constraints).
"""
import re

import numpy as np

FT = 0.3048
IN = 0.0254


def _clean(text):
    """Fix typical OCR confusions, only next to digits."""
    t = text.strip().replace("\u2019", "'").replace("\u2032", "'").replace("\u201d", '"').replace("\u2033", '"')
    for _ in range(3):                                  # 5.O0 -> 5.00, 1OO -> 100
        t = re.sub(r"(?<=[\d.])[Oo](?=[\d.]|\s|$)", "0", t)
    t = re.sub(r"(?<=\d)[lI|](?=\d)", "1", t)          # 1l5 -> 115
    t = t.replace(",", ".") if re.search(r"\d,\d{1,2}(?!\d)", t) else t.replace(",", "")
    return t


def parse_dimension(text):
    """Return {"metres", "unit", "confident"} or None if the text is not a dimension.
    confident=False means the unit was guessed (no unit written)."""
    if not text:
        return None
    t = _clean(text)

    # feet-inches:  12'6"   12' 6   12'-6"   12'   6"
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*'\s*-?\s*(?:(\d+(?:\.\d+)?)\s*(?:\"|in)?)?\s*", t)
    if m:
        feet, inch = float(m.group(1)), float(m.group(2) or 0)
        return {"metres": feet * FT + inch * IN, "unit": "ft-in", "confident": True}
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:\"|in)\s*", t)
    if m:
        return {"metres": float(m.group(1)) * IN, "unit": "in", "confident": True}

    # number + optional unit
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|M)?\s*", t)
    if not m:
        return None
    val, unit = float(m.group(1)), (m.group(2) or "").lower()
    if unit == "mm":
        return {"metres": val / 1000, "unit": "mm", "confident": True}
    if unit == "cm":
        return {"metres": val / 100, "unit": "cm", "confident": True}
    if unit == "m":
        return {"metres": val, "unit": "m", "confident": True}

    # no unit written: guess from the shape of the number
    if "." in m.group(1) and val < 40:
        return {"metres": val, "unit": "m?", "confident": False}
    if float(val).is_integer():
        if 1000 <= val <= 40000:
            return {"metres": val / 1000, "unit": "mm?", "confident": False}
        if 100 <= val < 1000:
            return {"metres": val / 100, "unit": "cm?", "confident": False}
    return None


def _point_to_segment(pt, a, b):
    """Distance from pt to segment ab, and where along ab (metres) the closest point is."""
    pt, a, b = np.array(pt, float), np.array(a, float), np.array(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.hypot(*(pt - a))), 0.0
    t = float(np.clip((pt - a) @ ab / L2, 0, 1))
    return float(np.hypot(*(pt - (a + t * ab)))), t * np.sqrt(L2)


def link_to_walls(ocr_items, plan, max_dist=1.2, max_rel_dev=0.4, min_confident=False):
    """Attach each readable dimension to the wall it describes.
    A dimension line is drawn beside its wall, so we look for walls within max_dist
    of the text, then pick the one whose length best matches the number."""
    walls = plan["walls"]
    constraints, skipped = [], []
    for item in ocr_items:
        d = parse_dimension(item["text"])
        if d is None:
            continue  # room names, notes, etc.
        if min_confident and not d["confident"]:
            skipped.append({**item, "reason": "unit guessed"})
            continue
        best = None
        for w in walls:
            dist, _ = _point_to_segment(item["center"], w["p1"], w["p2"])
            if dist > max_dist:
                continue
            L = float(np.hypot(w["p2"][0] - w["p1"][0], w["p2"][1] - w["p1"][1]))
            dev = abs(d["metres"] - L) / max(L, 1e-6)
            score = dev + 0.2 * dist       # prefer a close wall with a similar length
            if best is None or score < best[0]:
                best = (score, w["id"], dev)
        if best is None or best[2] > max_rel_dev:
            skipped.append({**item, "reason": "no wall with a plausible length nearby"})
            continue
        constraints.append({"wall": best[1], "length": round(d["metres"], 4),
                            "text": item["text"], "unit": d["unit"]})
    return constraints, skipped


if __name__ == "__main__":
    for s in ["5.00 m", "3,50m", "350 cm", "4200", "12'6\"", "12' 6", "9'-0\"", "5.O0 m", "LIVING", "2.4"]:
        print(f"{s!r:12} -> {parse_dimension(s)}")
