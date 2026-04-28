# Feature Tracking and Dynamic Object Detection

Computer Vision mid-course project (April 2026). Detects and localises
a moving object in the first frame of an RGB image sequence using
**classical CV only** (OpenCV, no deep learning).

## Pipeline

The brief recommends "sparse local features + robust feature matching
or **optical flow**". We follow the optical-flow path:

1. **CLAHE preprocessing** – local histogram equalisation
   (`cv::createCLAHE`, clip 0.5, 8×8 tiles) on every frame so the
   next stage can find corners on smooth/dim objects (the frog).
2. **Shi–Tomasi corners on frame 0** – `cv::goodFeaturesToTrack`
   (maxCorners 1500, qualityLevel 0.005, minDistance 5, blockSize 7).
3. **Pyramidal Lucas–Kanade tracking** – `cv::calcOpticalFlowPyrLK`
   propagates each corner frame-by-frame (window 21×21, 3 levels)
   with a forward–backward consistency filter (max round-trip drift
   1.5 px).
4. **Median-flow subtraction** – per frame, the median displacement
   across surviving tracks is subtracted to cancel constant camera
   motion (covers the slight car pan).
5. **Dynamic classification** – average residual displacement per
   track; keep the top 15% above `max(8 px, 4 × median)`. The floor
   rejects swaying leaves, distant walkers, and near-static objects
   (e.g. the almost-static sheep) as required by the brief.
6. **Clustering + bbox** – DBSCAN-like proximity clustering, neighbour
   merging, MAD outlier rejection at 2.5σ, padded bounding box.
7. **Silhouette refinement (gated)** – phase-correlation warped frame
   differencing + Otsu + morphology + largest connected component,
   accepted only when meaningfully smaller than the coarse box.

The first iteration used SIFT + BFMatcher + Lowe's ratio test in
place of stages 1–2; it underperformed on small/textureless objects
(squirrel IoU 0.31, sub-threshold) which is why we switched to the
optical-flow front-end.

## Results on the provided dataset

| category | IoU | TP@0.5 |
|----------|-----|--------|
| bird     | 0.578 | yes |
| car      | 0.773 | yes |
| frog     | 0.532 | yes |
| sheep    | 0.796 | yes |
| squirrel | 0.616 | yes |

**mIoU = 0.659, accuracy@0.5 = 1.00 (5/5).**

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
./build/detect dataset output
```

Expects the DAVIS/SegTrack2-derived dataset under `dataset/data/<cat>`
and `dataset/labels/<cat>/0000.txt`.

Outputs per category:
- `output/<cat>_pred.txt` – predicted `xmin ymin xmax ymax`
- `output/<cat>_overlay.png` – first frame with tracks + GT + prediction
- `output/<cat>_boxes.png` – first frame with GT + prediction only
- `output/<cat>_tracks.png` – first frame with tracks only
- `output/results.csv`, `output/summary.txt`

## Report

LaTeX source: [report/report.tex](report/report.tex). Compile with
`tectonic report/report.tex` or `pdflatex`.

## Source layout

```
include/dod.hpp
src/
  feature_tracker.cpp   -- Shi-Tomasi + Lucas-Kanade + forward-backward + median flow
  motion_segmenter.cpp  -- displacement-percentile dynamic classifier
  bbox_estimator.cpp    -- clustering + MAD + bbox + per-cluster helper
  silhouette_refiner.cpp -- phase-correlation warped absdiff + Otsu refine
  evaluator.cpp         -- IoU, GT parsing
  main.cpp              -- CLI entry point
CMakeLists.txt
```

## Dependencies

- OpenCV 4 (tested with 4.13.0)
- CMake 3.10+
- C++17 compiler

## Authors and file ownership

| File | Owner |
|------|-------|
| `include/dod.hpp` | Riccardo Pesce |
| `src/feature_tracker.cpp` | Mahdi Gheysari |
| `src/motion_segmenter.cpp` | Filippo Businaro |
| `src/bbox_estimator.cpp` | Riccardo Pesce |
| `src/silhouette_refiner.cpp` | Filippo Businaro |
| `src/evaluator.cpp` | Filippo Businaro |
| `src/main.cpp` | Mahdi Gheysari |

Design, tuning, and report writing were shared across the team.

## Reports

- [report/report.pdf](report/report.pdf) — 2-page required report.
- [report/explainer.pdf](report/explainer.pdf) — detailed method and
  code walk-through with diagrams.

## License / attribution

All source code written by the authors (Mahdi Gheysari, Filippo Businaro,
Riccardo Pesce). Dataset is derived from the publicly available DAVIS and
SegTrack2 datasets.
