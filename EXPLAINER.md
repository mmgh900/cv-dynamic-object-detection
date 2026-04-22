# What the hell is this project?

A plain-English walk-through of what the project does, why each step
exists, and what to look at when reading the code.

---

## 1. The problem

You get a short video of a scene (a folder of numbered PNG/JPG frames).
Somewhere in that video, **one thing is moving** (a bird, a car, a
frog, a sheep, or a squirrel). Your job is to look at the **first
frame** and draw a box around that moving thing.

Constraints:

- classical computer vision only — no deep learning,
- lighting can change across the sequence,
- objects can be small (the squirrel is ~23×31 pixels),
- parts of the object can be temporarily occluded.

Input: a folder of frames.
Output: `xmin ymin xmax ymax` for frame 0, plus a picture of the first
frame with the box drawn on it.

## 2. What we actually see in the data

Before picking a method we looked at each sequence:

- **bird, frog, sheep, squirrel** — camera is essentially static.
- **car** — camera does a small left-to-right pan, but the scene
  barely translates.

So the honest model is: *"the background barely moves; the object
does."* We don't need a full epipolar / homography machinery to
undo camera motion — we just need to remove any constant background
drift.

## 3. The core idea in one sentence

> **Detect the same interesting points in every frame, match them
> back to frame 0, subtract off the global background flow, and keep
> the points whose remaining displacement is unusually large. Those
> points sit on the moving object.**

## 4. The pipeline, stage by stage

### Stage 1 — Find interesting points on every frame  (`src/feature_tracker.cpp`)

We use **SIFT** via `cv::SIFT::create(1500)` and its
`detectAndCompute`. SIFT gives us up to 1500 keypoints per frame plus
a **128-D descriptor** per keypoint. SIFT is scale-invariant,
rotation-invariant, and reasonably illumination-invariant thanks to
descriptor normalisation — exactly what you want for outdoor
sequences.

### Stage 2 — Match frame 0 to every later frame (same file)

Between frame 0 and every frame `f > 0` we run a **brute-force L2
matcher** with `knnMatch(k=2)` — for each descriptor in frame 0, it
finds the **two** closest descriptors in frame `f`. We then apply
**Lowe's ratio test**: keep a match only if

```
best_distance < 0.75 * second_best_distance
```

The idea is simple: if the best and second-best matches are almost
equally good, the best match is probably ambiguous, so throw it out.
This is the canonical way to filter descriptor matches and is what
the course PDF recommends.

For every surviving match we record the displacement
`p_f - p_0` and accumulate, per frame-0 keypoint, a running total of
its motion.

### Stage 3 — Remove the global background drift (still `feature_tracker.cpp`)

Because the car sequence has a slight camera pan, all background
points drift in the same direction. We don't want to flag those as
dynamic. The fix is cheap: per frame, take the **median displacement
vector** across all matched points and subtract it from each point's
displacement before storing.

Static-camera sequences have near-zero median flow, so the
subtraction is harmless. Panning sequences have a non-zero median
that exactly cancels the camera component.

No homography, no RANSAC, no parametric model — just a median.

### Stage 4 — Decide which points are dynamic  (`src/motion_segmenter.cpp`)

For each frame-0 keypoint `i` we now have:

- `n_i` — how many frames had a successful match,
- `r_i` — the sum of relative displacement magnitudes across those
  frames.

Its score is `avg_i = r_i / n_i`, i.e. average per-frame residual
motion.

We compute the **median** of all scores and use it as the
"background motion scale". Then:

- **rank** all tracks by score, descending,
- **keep the top 15%** that also satisfy `score > max(6 px, 3 × median)`.

This relative cut-off adapts to the scene: a fully static camera has
a tiny median and almost anything above 6 px survives; a slight-pan
scene has a larger median and the 3× factor filters it out.

### Stage 5 — Turn dynamic points into a bounding box  (`src/bbox_estimator.cpp`)

We now have a cloud of suspicious dots on frame 0. Three problems:

- **outliers** — a few dynamic dots might be noise,
- **fragmentation** — on large objects the dots cluster in several
  blobs even though it's one object,
- **undercoverage** — SIFT lands on texture, not on the silhouette,
  so a tight box is usually smaller than the true object.

The fix:

1. **Cluster** with a DBSCAN-like proximity rule (radius ≈ 8% of the
   image diagonal, minPts = 3). Isolated noise dots are labelled as
   noise and ignored.
2. **Pick the heaviest cluster**, then **merge neighbouring clusters**
   whose boxes are within ~0.6× radius of it. This recovers objects
   whose texture fragments into parts (e.g.\ head vs body of a sheep).
3. **MAD outlier rejection** inside the merged cluster — drop any
   point more than `3 × median-absolute-deviation` from the cluster
   median in either axis.
4. **Pad the final box** by 1/6 of its size to cover silhouette
   pixels that SIFT tends to miss.

### Stage 6 — Score and write files  (`src/evaluator.cpp`, `src/main.cpp`)

For each category we:

- read the ground-truth `0000.txt` (four integers),
- compute **IoU** between predicted and GT boxes,
- mark a **true positive** if IoU > 0.5,
- write prediction to `output/<cat>_pred.txt`,
- save overlay PNGs (tracks, boxes, full).

At the end `main.cpp` prints **mIoU** and **accuracy@0.5**.

## 5. Why these particular choices?

- **Why SIFT + Lowe ratio and not KLT?**
  The PDF explicitly recommends "sparse local features + robust
  feature matching", and the course covers feature matching rather
  than Lucas-Kanade flow. SIFT + Lowe is the canonical classroom
  recipe and handles scale / illumination changes naturally.
- **Why median subtraction and not RANSAC homography?**
  Four of five sequences have a static camera and the fifth has only
  a slight translation. A full homography is overkill and its
  reprojection residuals are noisier than the displacement itself.
  The median vector cancels a constant translation exactly.
- **Why top-15% and a relative floor instead of a fixed threshold?**
  Fixed thresholds fail across scenes: static-camera bird/frog have
  tiny displacements everywhere, panning car has larger ones. A
  percentile plus `3 × median` floor self-adapts.
- **Why cluster-merging?**
  Without it the sheep box collapses to a single sheep.

## 6. Results we actually got

| category | GT box (w×h) | predicted (w×h) | IoU | TP@0.5 |
|----------|--------------|------------------|-----|--------|
| bird     | 283×255      | 329×249          | 0.545 | yes |
| car      | 68×42        | 77×42            | 0.783 | yes |
| frog     | 133×98       | 157×94           | 0.754 | yes |
| sheep    | 321×179      | 296×126          | 0.355 | no  |
| squirrel | 23×31        | 76×52            | 0.160 | no  |

- **mIoU = 0.519**
- **accuracy@0.5 = 0.60 (3/5)**

Failure modes:

- **squirrel** — tiny object (23×31 px) with low-texture fur; SIFT
  finds few stable extrema on it, a handful of mismatches on nearby
  leaves enlarge the box.
- **sheep** — multiple sheep plus a wire fence produce very
  repetitive descriptors, so Lowe's ratio rejects many legitimate
  matches and the cluster is under-covered.

Both failures are classical hard cases for sparse-descriptor
methods.

## 7. How to read the code

Suggested order:

1. `include/dod.hpp` — structs and signatures.
2. `src/main.cpp` — orchestration, one sequence at a time.
3. `src/feature_tracker.cpp` — SIFT + BFMatcher + Lowe + median-flow
   subtraction. **The most interesting file.**
4. `src/motion_segmenter.cpp` — percentile-based dynamic
   classification.
5. `src/bbox_estimator.cpp` — clustering, MAD, box.
6. `src/evaluator.cpp` — trivial I/O + IoU.

## 8. How to run it

```bash
# one-time
cmake -S . -B build
cmake --build build -j

# dataset must live under ./dataset/{data,labels}/<category>/
./build/detect dataset output
```

Outputs land in `output/` (prediction txt, overlay PNGs, CSV,
summary text). Figures for the report are regenerated by:

```bash
python3 scripts/make_figures.py
```

## 9. TL;DR

1. Run SIFT on every frame.
2. Match frame 0 → frame `f` with BFMatcher + Lowe's ratio (0.75).
3. Subtract the per-frame median displacement (kills camera drift).
4. Average remaining residual per track; keep the top 15% above
   `max(6, 3 × median)`.
5. Cluster the survivors, merge neighbours, MAD-filter, pad, output a
   box.

That's the whole project.
