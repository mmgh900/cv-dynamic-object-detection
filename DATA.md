# Data — not in git

`dataset/` (~90 MB) and `*.zip` are git-ignored.

Expected layout: `dataset/data/<category>/` (frames) and `dataset/labels/<category>/0000.txt` (first-frame boxes); see README.
Source: provided by the Computer Vision course (Univ. of Padova, A.Y. 2025/26) for the mid-term project — derived from the public DAVIS (https://davischallenge.org) and SegTrack v2 datasets. The original download `data-20260422T085218Z-3-001.zip` sits in the repo folder on this Mac (ignored); unzip it into `dataset/`.
The `experiments/*/dataset` entries are symlinks to `../../dataset`.
