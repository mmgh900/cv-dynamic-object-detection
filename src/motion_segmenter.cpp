// pick the points that really move (not just camera).

#include "dod.hpp"
#include <algorithm>

namespace dod {
namespace {
constexpr int    kMinObservations = 3;
constexpr double kKeepFraction    = 0.15;
constexpr double kFloorPx         = 8.0;
constexpr double kFloorMedianMul  = 4.0;
}

std::vector<int> classifyDynamicTracks(const std::vector<TrackedPoint>& tracks) {
    struct Item { int idx; double score; };
    std::vector<Item> items;
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        if (t.n_matches < kMinObservations) continue;
        items.push_back({(int)i, t.total_disp / (double)t.n_matches});
    }
    if (items.empty()) return {};

    // median = how much background moves on average.
    std::vector<double> s;
    s.reserve(items.size());
    for (const auto& it : items) s.push_back(it.score);
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    double median = s[s.size() / 2];
    double floor  = std::max(kFloorPx, kFloorMedianMul * median);

    // sort big to small.
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b){ return a.score > b.score; });

    int target = std::max(6, (int)(items.size() * kKeepFraction));
    std::vector<int> dyn;
    for (int k = 0; k < target && k < (int)items.size(); ++k) {
        if (items[k].score < floor) break; // too small, stop.
        dyn.push_back(items[k].idx);
    }
    return dyn;
}

} // namespace dod
