// Author: Mahdi Gheysari
// Clusters dynamic keypoints on the first frame and produces a
// single bounding box enclosing the dominant cluster.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>
#include <queue>

namespace dod {

// Simple DBSCAN-like clustering by Euclidean proximity.
static std::vector<int> proximityCluster(const std::vector<cv::Point2f>& pts,
                                         float eps, int min_pts) {
    const int N = (int)pts.size();
    std::vector<int> labels(N, -1);
    int cur = 0;
    for (int i = 0; i < N; ++i) {
        if (labels[i] != -1) continue;

        std::vector<int> neighbors;
        for (int j = 0; j < N; ++j) {
            if (cv::norm(pts[i] - pts[j]) <= eps) neighbors.push_back(j);
        }
        if ((int)neighbors.size() < min_pts) {
            labels[i] = 0; // noise
            continue;
        }

        cur++;
        std::queue<int> q;
        for (int n : neighbors) q.push(n);
        labels[i] = cur;
        while (!q.empty()) {
            int k = q.front(); q.pop();
            if (labels[k] == 0) labels[k] = cur;
            if (labels[k] != -1) continue;
            labels[k] = cur;
            std::vector<int> kn;
            for (int j = 0; j < N; ++j)
                if (cv::norm(pts[k] - pts[j]) <= eps) kn.push_back(j);
            if ((int)kn.size() >= min_pts)
                for (int v : kn) q.push(v);
        }
    }
    return labels;
}

cv::Rect estimateBBox(const std::vector<TrackedPoint>& tracks,
                      const std::vector<int>& dynamic_idx,
                      const cv::Size& img_size) {
    if (dynamic_idx.empty()) return cv::Rect();

    std::vector<cv::Point2f> pts;
    std::vector<float> weights;
    pts.reserve(dynamic_idx.size());
    for (int i : dynamic_idx) {
        pts.push_back(tracks[i].first);
        int tot = tracks[i].inlier_count + tracks[i].outlier_count;
        float w = tot > 0 ? (float)tracks[i].outlier_count / (float)tot : 0.f;
        weights.push_back(w);
    }

    // Eps chosen relative to image diagonal so it generalises.
    float diag = std::sqrt((float)(img_size.width * img_size.width
                                 + img_size.height * img_size.height));
    float eps = std::max(20.f, diag * 0.08f);
    auto labels = proximityCluster(pts, eps, 3);

    // Score each cluster by summed outlier weight.
    std::map<int, float> score;
    std::map<int, int> count;
    std::map<int, cv::Rect> cbox;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] <= 0) continue;
        score[labels[i]] += weights[i];
        count[labels[i]] += 1;
    }
    if (score.empty()) {
        std::vector<cv::Point2f> all;
        for (int i : dynamic_idx) all.push_back(tracks[i].first);
        return cv::boundingRect(all);
    }

    // Build per-cluster bbox.
    std::map<int, std::vector<cv::Point2f>> cluster_pts;
    for (size_t i = 0; i < labels.size(); ++i)
        if (labels[i] > 0) cluster_pts[labels[i]].push_back(pts[i]);
    for (auto& kv : cluster_pts) cbox[kv.first] = cv::boundingRect(kv.second);

    // Pick the best cluster, then merge every other cluster whose
    // bounding box is close to (or overlaps with) the best one, so
    // that a single object whose keypoints are split across two
    // sub-clusters (e.g. head + body of a sheep) is recovered.
    int best = -1; float best_s = -1;
    for (auto& kv : score) {
        if (count[kv.first] < 3) continue;
        if (kv.second > best_s) { best_s = kv.second; best = kv.first; }
    }
    if (best < 0) {
        std::vector<cv::Point2f> all;
        for (int i : dynamic_idx) all.push_back(tracks[i].first);
        return cv::boundingRect(all);
    }

    std::vector<cv::Point2f> cluster = cluster_pts[best];
    cv::Rect seed = cbox[best];
    float merge_gap = eps * 0.6f;
    for (auto& kv : cluster_pts) {
        if (kv.first == best) continue;
        cv::Rect r = cbox[kv.first];
        cv::Rect expanded(seed.x - (int)merge_gap, seed.y - (int)merge_gap,
                          seed.width + 2 * (int)merge_gap,
                          seed.height + 2 * (int)merge_gap);
        if ((r & expanded).area() > 0 && count[kv.first] >= 2) {
            cluster.insert(cluster.end(), kv.second.begin(), kv.second.end());
        }
    }

    // Reject outliers within the cluster using median absolute
    // deviation. Classical CV technique for robust spread estimation.
    if (cluster.size() >= 6) {
        std::vector<float> xs, ys;
        xs.reserve(cluster.size()); ys.reserve(cluster.size());
        for (const auto& p : cluster) { xs.push_back(p.x); ys.push_back(p.y); }
        std::nth_element(xs.begin(), xs.begin()+xs.size()/2, xs.end());
        std::nth_element(ys.begin(), ys.begin()+ys.size()/2, ys.end());
        float mx = xs[xs.size()/2], my = ys[ys.size()/2];
        std::vector<float> dx, dy;
        dx.reserve(cluster.size()); dy.reserve(cluster.size());
        for (const auto& p : cluster) {
            dx.push_back(std::fabs(p.x - mx));
            dy.push_back(std::fabs(p.y - my));
        }
        std::nth_element(dx.begin(), dx.begin()+dx.size()/2, dx.end());
        std::nth_element(dy.begin(), dy.begin()+dy.size()/2, dy.end());
        float madx = std::max(6.f, dx[dx.size()/2]);
        float mady = std::max(6.f, dy[dy.size()/2]);
        std::vector<cv::Point2f> kept;
        for (const auto& p : cluster) {
            if (std::fabs(p.x - mx) <= 3.0f * madx &&
                std::fabs(p.y - my) <= 3.0f * mady) {
                kept.push_back(p);
            }
        }
        if (kept.size() >= 4) cluster = kept;
    }
    cv::Rect box = cv::boundingRect(cluster);

    // Pad box slightly to cover full object footprint (points are
    // typically detected on texture inside, not at silhouette).
    int pad_x = std::max(6, box.width / 6);
    int pad_y = std::max(6, box.height / 6);
    box.x -= pad_x; box.y -= pad_y;
    box.width += 2 * pad_x; box.height += 2 * pad_y;
    box &= cv::Rect(0, 0, img_size.width, img_size.height);
    return box;
}

} // namespace dod
