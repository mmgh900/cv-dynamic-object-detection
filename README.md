# Feature Tracking and Dynamic Object Detection

Computer Vision mid-course project (April 2026). Detects and localises
a moving object in the first frame of an RGB image sequence using
**classical CV only** (OpenCV, no deep learning).

## Pipeline

1. **Sparse features** – Shi–Tomasi corners with adaptive min-distance.
2. **KLT tracking** – pyramidal Lucas–Kanade optical flow with
   forward-backward consistency check.
3. **Global motion** – RANSAC homography per consecutive frame pair.
4. **Dynamic classification** – top-15% tracks by (outlier ratio × mean
   reprojection residual).
5. **Clustering + bbox** – DBSCAN-like proximity clustering with
   neighbouring-cluster merging, then a padded bounding box.

## Results on the provided dataset

| category | IoU | TP@0.5 |
|----------|-----|--------|
| bird     | 0.636 | yes |
| car      | 0.576 | yes |
| frog     | 0.598 | yes |
| sheep    | 0.651 | yes |
| squirrel | 0.310 | no  |

**mIoU = 0.554, accuracy = 0.80 (4/5).**

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
  feature_tracker.cpp   -- Shi-Tomasi + KLT
  motion_segmenter.cpp  -- RANSAC homography, dynamic classification
  bbox_estimator.cpp    -- clustering + bbox
  evaluator.cpp         -- IoU, GT parsing
  main.cpp              -- CLI entry point
CMakeLists.txt
```

## Dependencies

- OpenCV 4 (tested with 4.13.0)
- CMake 3.10+
- C++17 compiler

## License / attribution

All source code written by the author (Mahdi Gheysari). Dataset is
derived from the publicly available DAVIS and SegTrack2 datasets.
