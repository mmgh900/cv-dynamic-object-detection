# What the hell is this project?

A plain-English walk-through of what the project does, why each step
exists, and what to look at when reading the code.

---

## 1. The problem

You get a short video of a scene (as a folder of numbered PNG/JPG
frames). Somewhere in that video, **one thing is moving** (a bird, a
car, a frog, a sheep, or a squirrel). Your job is to look at the
**first frame** and draw a box around that moving thing.

That's the whole task. Simple to describe, surprisingly annoying to
actually do, because:

- the camera might be moving too (shaking, panning),
- lighting changes across the sequence,
- the object might be small (the squirrel is ~23×31 pixels),
- parts of the object can be occluded,
- and we can't use deep learning — only classical computer vision.

Input: a folder of frames.
Output: `xmin ymin xmax ymax` for frame 0, plus a picture of the
first frame with the box drawn on it.

## 2. The core idea in one sentence

> **If we track image features across the sequence, almost everything
> will move in a way that's consistent with a single camera motion —
> except the features that sit on the moving object. Those are our
> clues.**

So the algorithm is basically:

1. Put dots on "interesting" pixels in frame 0.
2. Follow each dot through the video.
3. For each pair of frames, ask "can one global transformation explain
   how most dots moved?" — the transformation that explains the
   **majority** is the camera/background motion.
4. The dots that **don't** fit that transformation are suspicious.
   Those are our moving-object dots.
5. Group the suspicious dots in frame 0 and draw a box around them.

That's it. Everything else is implementation detail.

## 3. The pipeline, stage by stage

### Stage 1 — Put dots on the first frame  (`src/feature_tracker.cpp`)

We use **Shi–Tomasi corners** (`cv::goodFeaturesToTrack`). Corners are
good because they're locally unique — a feature tracker can tell them
apart from the patch next to them. A flat wall is useless; the corner
of a brick is great.

Two small tricks:
- **Sub-pixel refinement** (`cornerSubPix`) — pushes the corner
  coordinate to a fractional pixel. Helps tracking accuracy later.
- **Adaptive minimum distance** between corners, proportional to the
  image diagonal. Small images (like the squirrel sequence, 259×327)
  get denser sampling so small objects still catch enough corners.

Output: a few hundred to ~1500 `Point2f` seeds.

### Stage 2 — Follow each dot through the sequence (still `feature_tracker.cpp`)

This is the **Kanade–Lucas–Tomasi (KLT) tracker**, via
`cv::calcOpticalFlowPyrLK`. For each dot in frame `t`, it guesses
where it went in frame `t+1` by assuming the dot's local brightness
pattern barely changed (the "brightness-constancy" assumption) and
that motion was small. The "pyramidal" part means it does this at
several scales so it can handle bigger motions.

To kill bad tracks early, we do a **forward–backward consistency
check**: track forward from `t` → `t+1`, then backward from `t+1` →
`t`, and check you end up where you started (within ~1 pixel). If
not, drop the dot. This catches dots that drift onto the wrong thing,
get occluded, or hit a boundary.

Output: for each dot, a list of `(x, y)` positions across frames — a
**trajectory**.

### Stage 3 — Separate "moves with the world" from "moves on its own" (`src/motion_segmenter.cpp`)

This is the heart of the method.

Between every pair of consecutive frames, we ask:
> "Is there a single **homography** (a 3×3 projective transformation)
> that explains how most dots moved?"

We fit that homography using **RANSAC** (`cv::findHomography` with
`cv::RANSAC`). RANSAC repeatedly picks random small subsets of dots,
fits a homography to them, and counts how many other dots agree with
it ("inliers"). The best homography is the one with the most inliers.

Why a homography? Because if the scene is roughly planar, or the
camera is mostly rotating, a homography is a really good model for
**pure background motion**. Whatever doesn't fit the homography is
almost certainly moving independently.

For each dot we then accumulate two numbers over the whole sequence:
- `out_ratio` — fraction of frame-pairs where it was a RANSAC
  outlier (didn't fit the homography).
- `avg_res` — average distance between where the homography predicted
  it would be and where it actually ended up.

We combine those into one score:
```
score = out_ratio * min(avg_res, 50)
```
This says: "the more often you disagree with the global motion AND
the bigger your disagreement, the more suspicious you are."

We keep the **top 15%** of dots by this score, instead of using a
fixed threshold. Why the top 15%? Because scenes vary — sometimes the
camera is static and the object stands out hugely, sometimes the
camera pans and residuals are all bigger. A relative cut-off is more
robust than a magic number.

Output: indices of the dots classified as **dynamic**.

### Stage 4 — Turn dynamic dots into a bounding box (`src/bbox_estimator.cpp`)

Now we have a cloud of "suspicious" dots scattered on the first frame.
We need to return a single rectangle. Three problems:

- **Outliers.** A few dynamic dots might be noise, far from the real
  object.
- **Fragmentation.** On a large object (the sheep sequence has three
  sheep together), the dots cluster in several blobs even though it's
  one thing we want to box.
- **Undercoverage.** Corners tend to land on the **texture** of the
  object, not on its silhouette, so a tight box around the dots is
  usually smaller than the true object.

Our fix:
1. **Cluster the dots** with a DBSCAN-like proximity rule (radius
   ~8% of the image diagonal). This naturally drops isolated noise
   dots (they become "noise" with label 0).
2. **Pick the best cluster** (highest total outlier-weight) and then
   **merge neighbouring clusters** whose bounding boxes are within
   ~1.2× radius. This is what recovers the full sheep group instead
   of only boxing one sheep.
3. **Pad the final bounding box** by 1/8 of its size to compensate for
   the "corners land on texture, not silhouette" issue.

Output: a single `cv::Rect` in first-frame coordinates.

### Stage 5 — Score and write files (`src/evaluator.cpp`, `src/main.cpp`)

For each category we:
- read the ground-truth `0000.txt` (four integers),
- compute **IoU** between predicted and GT boxes,
- mark it a **true positive** if IoU > 0.5,
- write the prediction to `output/<cat>_pred.txt`,
- save an overlay PNG with tracks + GT (green) + prediction (red).

At the end, `main.cpp` prints **mean IoU** and **accuracy** across all
five categories.

## 4. Why these particular choices?

- **Why Shi–Tomasi + KLT instead of SIFT + matching?**
  KLT is way faster and gives continuous trajectories, which we need
  for the "how often does this dot disagree" signal. SIFT would give
  matches between distant frames but not a smooth trajectory, so we'd
  lose the temporal aggregation that makes the method robust.
- **Why homography and not fundamental matrix?**
  Fundamental matrix handles general 3D scenes with a moving camera
  but needs many more inliers and is noisier on short baselines.
  Homography is strictly a planar/rotational model but in practice
  works very well as a **dominant motion** approximation for these
  short clips.
- **Why a relative top-k and not a fixed threshold on residuals?**
  Tried fixed ("out_ratio > 0.45"): mIoU 0.42. Tried relative
  (top-15%): mIoU 0.56. Scenes are too different for one number.
- **Why cluster-merging?**
  Without it, sheep IoU was 0.36 (only one sheep boxed). With it, 0.65.

## 5. Results we actually got

| category | GT box (w×h) | predicted (w×h) | IoU | TP@0.5 |
|----------|--------------|------------------|-----|--------|
| bird     | 283×255      | 315×350          | 0.64 | yes |
| car      | 68×42        | 84×59            | 0.58 | yes |
| frog     | 133×98       | 101×90           | 0.60 | yes |
| sheep    | 321×179      | 224×172          | 0.65 | yes |
| squirrel | 23×31        | 15×20            | 0.31 | **no** |

- **mIoU = 0.554**
- **accuracy = 0.80 (4/5)**

Squirrel is the one failure: it's a ~23×31 px object in a portrait
image with tiny inter-frame motion. Only 6 dots end up classified as
dynamic, so the resulting box is too small. The centre is roughly
right, but the area isn't.

Possible improvements (all still classical CV):
- re-sample features densely around candidate dynamic regions,
- expand the final box using colour / edge agreement (GrabCut,
  SLIC superpixels, simple edge-density growing).

## 6. How to read the code

If you're going to read only one file, make it
**`src/motion_segmenter.cpp`** — that's where the interesting idea
lives.

Suggested order:
1. `include/dod.hpp` — look at the structs and function signatures,
   you get the whole data flow in 30 seconds.
2. `src/main.cpp` — top-level orchestration, one sequence at a time.
3. `src/feature_tracker.cpp` — detection + KLT.
4. `src/motion_segmenter.cpp` — the homography / outlier logic.
5. `src/bbox_estimator.cpp` — clustering and the final rectangle.
6. `src/evaluator.cpp` — trivial I/O and IoU.

## 7. How to run it

```bash
# one-time
cmake -S . -B build
cmake --build build -j

# dataset must live under ./dataset/{data,labels}/<category>/
./build/detect dataset output
```

Outputs land in `output/` (prediction txt, overlay PNG, tracks PNG,
results CSV, summary text).

Figures for the report are regenerated from `output/` by:

```bash
python3 scripts/make_figures.py
```

## 8. What's in each folder

```
mid-project/
├── CMakeLists.txt          build recipe
├── include/dod.hpp         shared declarations
├── src/
│   ├── feature_tracker.cpp stage 1 + 2
│   ├── motion_segmenter.cpp stage 3
│   ├── bbox_estimator.cpp  stage 4
│   ├── evaluator.cpp       GT / IoU
│   └── main.cpp            entry point
├── scripts/make_figures.py  figure generator
├── dataset/                 (gitignored) input frames + GT
├── output/                  per-run predictions and visualisations
└── report/
    ├── report.tex           2-page LaTeX report
    ├── report.pdf           compiled version
    └── figures/             figures used by the report
```

## 9. TL;DR

1. Drop corners on frame 0.
2. Track them with KLT.
3. For each pair of frames, find the global motion with RANSAC
   homography.
4. The dots that consistently don't fit the global motion are on the
   moving object.
5. Cluster them, merge nearby clusters, pad a bit, output a box.

That's the whole project.
