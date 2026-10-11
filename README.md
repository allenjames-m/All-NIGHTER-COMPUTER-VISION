# All-Nighter Computer Vision: floor plan to trustworthy 3D (HNX26EPS06, Mode A)

Turns a floor plan into a 3D model (GLB) and says which parts were read from the plan, which were
corrected from written dimensions, and which are assumed defaults.

**Colours in the 3D viewer:** green = detected from the plan, yellow = corrected or inferred, red = assumed default.

## Pipeline
```
plan image -> detector -> scale from dimension text -> dimension solver -> validation -> 3D (GLB)
              (baseline)   dims.py + solver.py          solver.py          validate.py   build_3d.py
```

## Quick start (Windows PowerShell, Python 3.12+)
```powershell
pip install -r requirements.txt
python pipeline/run_pipeline.py data/demo_detected.json data/demo_ocr.json --gt data/demo_gt.json
cd viewer
python -m http.server 8000 --bind 127.0.0.1      # then open http://localhost:8000
```

## Reproduce the evaluation
```powershell
# 1. ground truth + test images from CubiCasa5K (official test split, 30 plans)
python pipeline/svg_to_plan.py --batch "C:\path\to\cubicasa5k" --split test.txt --n 30 --out data
python pipeline/stamp_dimensions.py --all data      # the dataset has no written dimensions; we draw the true ones
# 2. tests
python eval/test_metrics.py ; python eval/test_dims.py ; python eval/test_validate.py ; python eval/test_solver.py
# 3. ablation table (simulated detector)
python eval/ablation.py data
# 3b. same table for a real detector: one JSON per plan in preds/, same coordinate frame as data/stamped/*.png
python eval/ablation.py data --pred-dir preds
```

## Files
| File | Purpose |
|---|---|
| `pipeline/build_3d.py` | plan JSON to coloured GLB (walls cut around doors/windows, floors from rooms) |
| `pipeline/dims.py` | reads dimension text ("5.00 m", "12'6\"") and links each number to the right wall |
| `pipeline/solver.py` | scale from text and the axis-line dimension solver |
| `pipeline/validate.py` | thickness-aware consistency checks; fixes only noise-sized gaps, reports the rest |
| `pipeline/run_pipeline.py` | one command through all stages, scoring each against ground truth |
| `pipeline/svg_to_plan.py`, `stamp_dimensions.py` | dataset to ground truth and dimension-labelled images |
| `eval/metrics.py`, `eval/ablation.py` | layout IoU, dimension error, door/window F1, ablation table |

## Plan JSON format (metres)
`walls[{id,p1,p2,source}]`, `doors[{wall,pos,width,source}]`, `windows[...]`, `rooms[{name,polygon}]`;
`source` is `detected`, `inferred` or `assumed`.

## Limitations (read these)
See `WRITEUP.md`. Short version: the headline table uses a **simulated detector** and **perfect OCR** on **synthetically
stamped dimensions**. It measures whether the dimension steps help; it is not a comparison against a real parser.
