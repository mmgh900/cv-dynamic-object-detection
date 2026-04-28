// group the moving points, find the biggest group, draw a box.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <map>

namespace dod {
namespace {
constexpr float kEpsFraction = 0.08f;
constexpr float kEpsMinPx    = 20.f;
constexpr int   kMinPts      = 3;
constexpr float kMergeFactor = 0.6f;
constexpr float kMadK        = 2.5f;
constexpr float kMadFloor    = 6.f;
constexpr int   kPadDivisor  = 6;
constexpr int   kPadMin      = 6;
}

// simple DBSCAN-like grouping by distance.
static std::vector<int> proximityCluster(const std::vector<cv::Point2f>& pts,
                                         float eps, int min_pts) {
    int N = (int)pts.size();
    std::vector<int> labels(N, -1);
    int cur = 0;
    auto neighborsOf = [&](int i) {
        std::vector<int> nb;
        for (int j = 0; j < N; ++j)
            if (cv::norm(pts[i] - pts[j]) <= eps) nb.push_back(j);
        return nb;
    };
    for (int i = 0; i < N; ++i) {
        if (labels[i] != -1) continue;
        auto nb = neighborsOf(i);
        if ((int)nb.size() < min_pts) { labels[i] = 0; continue; }
        labels[i] = ++cur;
        std::queue<int> q;
        for (int n : nb) q.push(n);
        while (!q.empty()) {
            int k = q.front(); q.pop();
            if (labels[k] == 0) labels[k] = cur;
            if (labels[k] != -1) continue;
            labels[k] = cur;
            auto kn = neighborsOf(k);
            if ((int)kn.size() >= min_pts)
                for (int v : kn) q.push(v);
        }
    }
    return labels;
}

static float medianFloor(std::vector<float>& v, float floor) {
    auto m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    return std::max(floor, v[m]);
}

cv::Rect estimateBBox(const std::vector<TrackedPoint>& tracks,
                      const std::vector<int>& dynamic_idx,
                      const cv::Size& img_size) {
    if (dynamic_idx.empty()) return {};

    std::vector<cv::Point2f> pts;
    pts.reserve(dynamic_idx.size());
    for (int i : dynamic_idx) pts.push_back(tracks[i].first);

    // eps depends on image size.
    float diag = std::sqrt((float)(img_size.width * img_size.width
                                 + img_size.height * img_size.height));
    float eps  = std::max(kEpsMinPx, diag * kEpsFraction);
    auto labels = proximityCluster(pts, eps, kMinPts);

    std::map<int, std::vector<cv::Point2f>> by_label;
    for (size_t i = 0; i < labels.size(); ++i)
        if (labels[i] > 0) by_label[labels[i]].push_back(pts[i]);
    if (by_label.empty()) return cv::boundingRect(pts);

    // biggest group wins.
    int best = -1; size_t best_n = 0;
    for (auto& kv : by_label)
        if (kv.second.size() > best_n) { best_n = kv.second.size(); best = kv.first; }

    std::vector<cv::Point2f> cluster = by_label[best];
    cv::Rect seed = cv::boundingRect(cluster);

    // also take groups that touch the winner (e.g. head + body).
    int gap = (int)(eps * kMergeFactor);
    cv::Rect expanded(seed.x - gap, seed.y - gap,
                      seed.width + 2 * gap, seed.height + 2 * gap);
    for (auto& kv : by_label) {
        if (kv.first == best) continue;
        cv::Rect r = cv::boundingRect(kv.second);
        if ((r & expanded).area() > 0)
            cluster.insert(cluster.end(), kv.second.begin(), kv.second.end());
    }

    // throw away points too far from the median.
    if (cluster.size() >= 6) {
        std::vector<float> xs, ys;
        for (const auto& p : cluster) { xs.push_back(p.x); ys.push_back(p.y); }
        std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
        std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
        float mx = xs[xs.size() / 2], my = ys[ys.size() / 2];
        std::vector<float> dx, dy;
        for (const auto& p : cluster) {
            dx.push_back(std::fabs(p.x - mx));
            dy.push_back(std::fabs(p.y - my));
        }
        float madx = medianFloor(dx, kMadFloor);
        float mady = medianFloor(dy, kMadFloor);
        std::vector<cv::Point2f> kept;
        for (const auto& p : cluster)
            if (std::fabs(p.x - mx) <= kMadK * madx &&
                std::fabs(p.y - my) <= kMadK * mady)
                kept.push_back(p);
        if (kept.size() >= 4) cluster = kept;
    }

    // box around the points + a bit of padding.
    cv::Rect box = cv::boundingRect(cluster);
    int padx = std::max(kPadMin, box.width  / kPadDivisor);
    int pady = std::max(kPadMin, box.height / kPadDivisor);
    box.x -= padx; box.y -= pady;
    box.width  += 2 * padx;
    box.height += 2 * pady;
    return box & cv::Rect(0, 0, img_size.width, img_size.height);
}

} // namespace dod
