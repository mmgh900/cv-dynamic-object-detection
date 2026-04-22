#!/usr/bin/env python3
"""Create supplementary figures for the report:
  - triptych of first/middle/last frame per category (with GT on first)
  - an IoU bar chart summary
Uses only numpy / PIL / matplotlib (no OpenCV).
"""
import os, csv, glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "data")
LBL  = os.path.join(ROOT, "dataset", "labels")
OUT  = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

CATS = ["bird", "car", "frog", "sheep", "squirrel"]

def list_frames(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")) +
                glob.glob(os.path.join(d, "*.jpg")))
    return fs

def read_gt(p):
    with open(p) as f:
        x1,y1,x2,y2 = map(int, f.read().split())
    return (x1,y1,x2,y2)

def read_pred(p):
    with open(p) as f:
        x1,y1,x2,y2 = map(int, f.read().split())
    return (x1,y1,x2,y2)

def triptych(cat):
    frames = list_frames(os.path.join(DATA, cat))
    if not frames: return
    picks = [frames[0], frames[len(frames)//2], frames[-1]]
    gt = read_gt(os.path.join(LBL, cat, "0000.txt"))
    pred_path = os.path.join(ROOT, "output", f"{cat}_pred.txt")
    pred = read_pred(pred_path) if os.path.exists(pred_path) else None

    imgs = [Image.open(p).convert("RGB") for p in picks]
    d0 = ImageDraw.Draw(imgs[0])
    d0.rectangle(gt, outline=(0,255,0), width=3)
    if pred:
        d0.rectangle(pred, outline=(255,0,0), width=3)

    w = max(im.width for im in imgs); h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (3*w + 20, h + 26), (255,255,255))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i*(w + 10), 20))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    labels = ["first (with GT green, pred red)", "middle", "last"]
    for i, t in enumerate(labels):
        d.text((i*(w+10)+6, 3), t, fill=(0,0,0), font=font)
    canvas.save(os.path.join(OUT, f"{cat}_triptych.png"))

def bar_chart():
    csv_path = os.path.join(ROOT, "output", "results.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    cats = [r["category"] for r in rows]
    ious = [float(r["iou"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.bar(cats, ious, color=["#4C78A8"]*5)
    for b, v in zip(bars, ious):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="IoU=0.5 threshold")
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-category IoU")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "iou_bars.pdf"))
    fig.savefig(os.path.join(OUT, "iou_bars.png"), dpi=150)

def copy_overlays():
    import shutil
    for cat in CATS:
        for suf in ("overlay", "tracks", "boxes"):
            src = os.path.join(ROOT, "output", f"{cat}_{suf}.png")
            dst = os.path.join(OUT, f"{cat}_{suf}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)

if __name__ == "__main__":
    for c in CATS:
        triptych(c)
    bar_chart()
    copy_overlays()
    print("figures generated in", OUT)
