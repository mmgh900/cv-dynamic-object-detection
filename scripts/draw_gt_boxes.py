"""Draw ground-truth bounding boxes on the first frame of each category.

Reads:  dataset/data/<cat>/0000.png  and  dataset/labels/<cat>/0000.txt
Writes: dataset/first_frame_boxes/<cat>_gt.png  +  all_gt.png (grid)

Label format: one line "x1 y1 x2 y2" (pixel coords, top-left / bottom-right).
"""

from pathlib import Path

import cv2
import numpy as np

CATEGORIES = ["bird", "car", "frog", "sheep", "squirrel"]
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data"
LABELS = ROOT / "dataset" / "labels"
OUT = ROOT / "dataset" / "first_frame_boxes"
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)


def load_bbox(path: Path) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in path.read_text().split())
    return x1, y1, x2, y2


def draw(img: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), BOX_COLOR, -1)
    cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return out


def make_grid(images: list[np.ndarray], cols: int = 3) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: im.shape[0], : im.shape[1]] = im
        padded.append(canvas)
    while len(padded) % cols != 0:
        padded.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = [np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)]
    return np.vstack(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overlays: list[np.ndarray] = []
    for cat in CATEGORIES:
        frame_candidates = sorted((DATA / cat).glob("0000.*"))
        if not frame_candidates:
            raise FileNotFoundError(DATA / cat / "0000.*")
        frame_path = frame_candidates[0]
        label_path = LABELS / cat / "0000.txt"
        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(frame_path)
        bbox = load_bbox(label_path)
        overlay = draw(img, bbox, cat)
        out_path = OUT / f"{cat}_gt.png"
        cv2.imwrite(str(out_path), overlay)
        overlays.append(overlay)
        print(f"{cat}: {bbox} -> {out_path.relative_to(ROOT)}")

    grid = make_grid(overlays, cols=3)
    grid_path = OUT / "all_gt.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"grid -> {grid_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
