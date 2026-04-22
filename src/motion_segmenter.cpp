// Author: Mahdi Gheysari
// Separates tracks that follow global camera motion (background)
// from tracks that belong to independently moving objects (foreground).

#include "dod.hpp"
#include <opencv2/calib3d.hpp>
#include <numeric>

namespace dod {

// For each consecutive frame pair, estimate a global homography from
// all alive tracks and count inlier/outlier votes per track. Tracks
// with many outlier votes are classified as dynamic.
std::vector<int> classifyDynamicTracks(const std::vector<cv::Mat>& frames_gray,
                                       std::vector<TrackedPoint>& tracks,
                                       double ransac_thresh) {
    const size_t n_frames = frames_gray.size();
    if (n_frames < 2) return {};

    // Reset vote counters.
    for (auto& t : tracks) {
        t.inlier_count = 0;
        t.outlier_count = 0;
        t.residual = 0.f;
    }

    for (size_t f = 1; f < n_frames; ++f) {
        std::vector<cv::Point2f> a, b;
        std::vector<int> idx;
        a.reserve(tracks.size());
        b.reserve(tracks.size());
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto& tr = tracks[i];
            if (tr.trajectory.size() > f) {
                a.push_back(tr.trajectory[f - 1]);
                b.push_back(tr.trajectory[f]);
                idx.push_back((int)i);
            }
        }
        if (a.size() < 8) continue;

        std::vector<uchar> mask;
        cv::Mat H = cv::findHomography(a, b, cv::RANSAC, ransac_thresh, mask, 2000, 0.995);
        if (H.empty()) continue;

        // Compute reprojection residuals to accumulate evidence.
        std::vector<cv::Point2f> a_proj;
        cv::perspectiveTransform(a, a_proj, H);
        for (size_t k = 0; k < idx.size(); ++k) {
            float dx = a_proj[k].x - b[k].x;
            float dy = a_proj[k].y - b[k].y;
            float r  = std::sqrt(dx * dx + dy * dy);
            tracks[idx[k]].residual += r;
            if (mask[k]) tracks[idx[k]].inlier_count++;
            else          tracks[idx[k]].outlier_count++;
        }
    }

    // Combine two signals: the outlier ratio against the global
    // homography and the average reprojection residual. We first keep
    // tracks with enough evidence, then take the top tracks by a
    // combined score so that we adapt to different scene dynamics
    // (static vs. moving camera) without hard thresholds.
    struct Item { int idx; double score; double avg_res; double out_ratio; int total; };
    std::vector<Item> items;
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        int total = t.inlier_count + t.outlier_count;
        if (total < 3) continue;
        double out_ratio = double(t.outlier_count) / double(total);
        double avg_res   = t.residual / std::max(1, total);
        double score     = out_ratio * std::min(avg_res, 50.0);
        items.push_back({(int)i, score, avg_res, out_ratio, total});
    }
    if (items.empty()) return {};

    // Sort descending by score and pick top tracks. Keep only those
    // with a non-trivial score (at least ransac_thresh * 0.3 residual
    // and >30% outlier fraction) — the algorithm backs off to looser
    // criteria when few candidates are available.
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b){ return a.score > b.score; });

    double top_quantile = 0.15;
    int n_keep = std::max(8, (int)(items.size() * top_quantile));
    n_keep = std::min(n_keep, (int)items.size());

    std::vector<int> dynamic_idx;
    for (int k = 0; k < n_keep; ++k) {
        if (items[k].avg_res < ransac_thresh * 0.3) break;
        if (items[k].out_ratio < 0.25) break;
        dynamic_idx.push_back(items[k].idx);
    }
    return dynamic_idx;
}

} // namespace dod
