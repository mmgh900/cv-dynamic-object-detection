#!/usr/bin/env python3
"""CLAHE before/after with detected Shi-Tomasi corners overlaid.
Shows why CLAHE rescued the frog: more corners land on the body.
"""
import os, glob, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "data")
OUT  = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

# Same parameters the C++ pipeline uses.
CLIP = 0.5
TILE = 8
CORNER_KW = dict(maxCorners=1500, qualityLevel=0.005,
                 minDistance=5, blockSize=7)

def detect_corners(gray):
    pts = cv2.goodFeaturesToTrack(gray, **CORNER_KW)
    return [] if pts is None else pts.reshape(-1, 2)

def histogram(gray, ax, title):
    ax.hist(gray.ravel(), bins=64, range=(0, 255),
            color="#3f6fb0", edgecolor="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

def overlay_corners(bgr, pts):
    out = bgr.copy()
    for p in pts:
        cv2.circle(out, (int(p[0]), int(p[1])), 3, (40, 80, 220), -1)
        cv2.circle(out, (int(p[0]), int(p[1])), 3, (255, 255, 255), 1)
    return out

def make_figure(category):
    frames = sorted(glob.glob(os.path.join(DATA, category, "*.png")) +
                    glob.glob(os.path.join(DATA, category, "*.jpg")))
    if not frames:
        return
    bgr = cv2.imread(frames[0])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE))
    gray_eq = clahe.apply(gray)
    bgr_eq  = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    pts_raw = detect_corners(gray)
    pts_eq  = detect_corners(gray_eq)

    raw_overlay = overlay_corners(bgr,   pts_raw)
    eq_overlay  = overlay_corners(bgr_eq, pts_eq)

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6),
                           gridspec_kw=dict(hspace=0.35, wspace=0.08))

    ax[0, 0].imshow(cv2.cvtColor(raw_overlay, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title(f"original  ({len(pts_raw)} corners)",
                       fontsize=10)
    ax[0, 0].axis("off")
    ax[0, 1].imshow(cv2.cvtColor(eq_overlay,  cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title(f"after CLAHE  ({len(pts_eq)} corners)",
                       fontsize=10)
    ax[0, 1].axis("off")

    histogram(gray,    ax[1, 0], "intensity histogram, original")
    histogram(gray_eq, ax[1, 1], "intensity histogram, after CLAHE")

    out = os.path.join(OUT, f"{category}_clahe.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out, "raw=", len(pts_raw), "eq=", len(pts_eq))

if __name__ == "__main__":
    for c in ["frog", "squirrel"]:
        make_figure(c)
