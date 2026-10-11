# Dimension-aware, honest floor-plan reconstruction (HNX26EPS06, Mode A)

## 1. Problem
Floor-plan parsers detect walls as pixels and extrude them. They ignore the dimensions written on the plan, so
rooms come out the wrong size, and they never say which parts were guessed. Our entry reads the plan, uses the
written dimensions to fix scale and wall positions, checks the result for consistency, and exports a GLB in which
every element is coloured by where its information came from.

## 2. Method
1. **Detector (baseline).** Any parser that outputs walls, doors, windows and rooms. Its output is our baseline.
2. **Scale from dimension text (`dims.py`, `solver.calibrate_scale`).** Written dimensions ("5.00 m") are parsed
   (metres, cm, mm, feet-inches, common OCR slips) and linked to the nearest wall with a plausible length.
   If the labels agree on a scale factor that differs from 1 by more than 3 %, the plan is rescaled.
3. **Axis-line dimension solver (`solver.solve_lines`).** All wall ends and room corners are grouped into shared
   x- and y-lines, then a robust least-squares solve moves the lines so labelled wall lengths match. Moving a line
   moves everything attached to it, so one dimension corrects walls and room corners together. Readings more than
   40 % from the detected length are rejected as OCR errors. Everything that moves is marked `inferred`.
4. **Validation (`validate.py`).** Thickness-aware checks: openings lie fully on their wall, no overlaps, windows on
   exterior walls, door borders a room, each room outline is backed by walls. Only noise-sized gaps (5 cm) are fixed;
   anything larger is reported, never silently changed.
5. **3D export (`build_3d.py`).** Walls are cut around doors and windows; floors follow room polygons; colours encode
   provenance (green detected, yellow inferred, red assumed defaults such as 2.7 m wall height).

## 3. Data and protocol
* 30 plans from the **official CubiCasa5K test split** (`high_quality_architectural`). Ground truth is converted
  from the dataset's SVG annotations. Scale (100 px per metre) comes from the hidden "W m x H m" room labels in the
  SVG, which agree within 5 % on at least 80 % of rooms for every plan used.
* CubiCasa images contain **no written dimensions**. We draw the true wall lengths onto copies of the images
  (`stamp_dimensions.py`). The dimension text is therefore **synthetic**.
* **OCR is perfect** in the ablation: label text and positions come from the stamping step. Real OCR mistakes are
  not yet measured.
* **The detector is simulated:** ground-truth geometry with every straight wall line shifted by about 8 cm
  (Gaussian), 1 cm corner jitter, 10 cm door shifts, and a random +/-6 % scale error. Five noise draws per plan.
  This tests whether the dimension steps help. **It is not a comparison against a real parser.**

## 4. Results (30 plans, simulated detector, perfect OCR)
| Stage | Layout IoU | Dim error (cm) |
|---|---|---|
| 0 Detector output | 0.828 | 20.1 |
| 1 + scale from text | 0.898 | 10.7 |
| 2 + snapping | 0.898 | 11.2 |
| 3 + dimension solver | 0.901 | 7.4 |
| 4 + validation | 0.901 | 7.5 |

Other settings, same protocol:

| Setting | Detector | Final | 
|---|---|---|
| No scale error (known resolution) | IoU 0.931, 9.3 cm | IoU 0.934, 6.6 cm |
| Larger line errors (15 cm) | IoU 0.810, 26.4 cm | IoU 0.854, 15.1 cm |

**What helps:** recovering scale from the written dimensions (large effect only when the scale is wrong) and the
dimension solver (about 3 cm less dimension error in every setting).
**What does not:** Manhattan snapping made dimension error 0.4 cm *worse* in every setting, so we do not claim it.
Validation does not change accuracy; it is a consistency and honesty layer.

## 5. Negative results and limits
* A classical image-processing detector we wrote as our own baseline reached a mean layout IoU of only 0.10 on
  these plans (no rooms found on 15 of 30). We did not use it. Walls drawn as thin double lines vanish, and gaps in
  outer walls let the interior leak to the outside.
* An earlier node-merging solver damaged real plans (thick walls do not meet at points). We replaced it with the
  axis-line solver above and verified that perfect plans pass through unchanged (IoU 1.000, 0.0 cm).
* Not yet done: comparison against a real off-the-shelf parser; real OCR; door and window detection from images;
  curved or diagonal walls (diagonal walls are left unchanged); plans without written dimensions (no gain possible).
* **[TODO before submission]** add the real-parser table: `python eval/ablation.py data --pred-dir preds`.

## 6. Honesty (provenance)
Every wall, door, window and room carries `source`. The viewer colours green / yellow / red. In the simulated runs,
the solver marks only the walls it actually moved (more than 1 cm) as inferred.
