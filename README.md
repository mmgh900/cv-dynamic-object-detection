# Feature Tracking and Dynamic Object Detection

Computer Vision mid-course project (April 2026). Detects and localises
a moving object in the first frame of an RGB image sequence using
**classical CV only** (OpenCV, no deep learning).

## Pipeline

1. **SIFT features** – `cv::SIFT::create(1500)` + `detectAndCompute`
   on every frame (128-D descriptors).
2. **Descriptor matching** – `cv::BFMatcher(NORM_L2)` with
   `knnMatch(k=2)` between frame 0 and every later frame, filtered by
   **Lowe's ratio test** at 0.75.
3. **Median-flow subtraction** – per frame, the median displacement
   across matched keypoints is subtracted from each match to cancel
   any constant camera translation (covers the slight car pan).
4. **Dynamic classification** – average residual displacement per
   track; keep the top 15% above `max(6 px, 3 × median)`.
5. **Clustering + bbox** – DBSCAN-like proximity clustering, neighbour
   merging, MAD outlier rejection, padded bounding box.

## Results on the provided dataset

| category | IoU | TP@0.5 |
|----------|-----|--------|
| bird     | 0.545 | yes |
| car      | 0.783 | yes |
| frog     | 0.754 | yes |
| sheep    | 0.355 | no  |
| squirrel | 0.160 | no  |

**mIoU = 0.519, accuracy@0.5 = 0.60 (3/5).**

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
  feature_tracker.cpp   -- SIFT + BFMatcher + Lowe ratio + median flow
  motion_segmenter.cpp  -- displacement-percentile dynamic classifier
  bbox_estimator.cpp    -- clustering + MAD + bbox
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
