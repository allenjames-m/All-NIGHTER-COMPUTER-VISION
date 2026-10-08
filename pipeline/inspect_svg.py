"""Print what is inside CubiCasa model.svg files.
Usage: python pipeline/inspect_svg.py C:\\Users\\user\\cubicasa5k
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
svgs = list(root.rglob("model.svg"))
print(f"Found {len(svgs)} model.svg files")
for svg in svgs[:3]:
    print("\n=====", svg.parent)
    tree = ET.parse(svg)
    counts = Counter()
    samples = {}
    for el in tree.iter():
        tag = el.tag.split("}")[-1]
        cls = el.get("class") or el.get("id") or ""
        key = f"{tag} | {cls[:50]}"
        counts[key] += 1
        if key not in samples:
            samples[key] = {k: v[:90] for k, v in el.attrib.items()}
    for key, n in counts.most_common(25):
        print(f"{n:4d}  {key}")
        print(f"      e.g. {samples[key]}")
