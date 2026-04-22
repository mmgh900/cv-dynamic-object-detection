// Author: Filippo Businaro
// Module owner: Filippo Businaro (dynamic-vs-static classification of tracks).
//
// Classifies tracked keypoints as dynamic (on a moving object) vs.
// static (background) using a displacement-percentile criterion.
//
// The dataset has a static or near-static camera (only the car
// sequence shows a slight pan), so the simplest honest model is:
// background keypoints have near-zero total displacement across the
// sequence, object keypoints have a large total displacement. We
// keep the top percentile of tracks by average per-frame displacement,
// above a robust absolute floor derived from the median.

#include "dod.hpp"
#include <algorithm>
#include <numeric>

namespace dod {

// The `ransac_thresh` argument is kept in the signature for API
// compatibility but is not used by this implementation.
std::vector<int> classifyDynamicTracks(const std::vector<cv::Mat>& frames_gray,
                                       std::vector<TrackedPoint>& tracks,
                                       double /*ransac_thresh*/) {
    (void)frames_gray;

    struct Item { int idx; double score; int obs; };
    std::vector<Item> items;
    items.reserve(tracks.size());
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        // inlier_count is repurposed: number of successful matches
        // residual          : sum of ||p_f - p_0|| over successful matches
        if (t.inlier_count < 3) continue;
        double avg = double(t.residual) / double(t.inlier_count);
        items.push_back({(int)i, avg, t.inlier_count});
    }
    if (items.empty()) return {};

    // Robust background-displacement scale from the median.
    std::vector<double> scores;
    scores.reserve(items.size());
    for (const auto& it : items) scores.push_back(it.score);
    std::vector<double> scores_copy = scores;
    std::nth_element(scores_copy.begin(),
                     scores_copy.begin() + scores_copy.size() / 2,
                     scores_copy.end());
    double median = scores_copy[scores_copy.size() / 2];

    // Top-percentile selection.
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b){ return a.score > b.score; });

    double floor_disp = std::max(6.0, 3.0 * median);
    int target = std::max(6, (int)(items.size() * 0.15));
    target = std::min(target, (int)items.size());

    std::vector<int> dynamic_idx;
    for (int k = 0; k < target; ++k) {
        if (items[k].score < floor_disp) break;
        dynamic_idx.push_back(items[k].idx);
    }

    // Tag selected tracks so downstream clustering can weight them.
    for (int i : dynamic_idx) tracks[i].outlier_count = 1;
    return dynamic_idx;
}

} // namespace dod
