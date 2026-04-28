// find corners on frame 0 and follow them with optical flow.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <filesystem>
#include <algorithm>

namespace dod {
namespace {
constexpr int    kMaxCorners      = 1500;
constexpr double kQualityLevel    = 0.005;
constexpr double kMinDistance     = 5.0;
constexpr int    kBlockSize       = 7;
constexpr int    kPyrWinSize      = 21;
constexpr int    kPyrMaxLevel     = 3;
constexpr float  kForwardBackErr  = 1.5f;
}

std::vector<std::string> listFrames(const std::string& folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(folder)) {
        std::string ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// median of x and y, separately.
static cv::Point2f medianXY(std::vector<cv::Point2f> v) {
    auto m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end(),
                     [](cv::Point2f a, cv::Point2f b){ return a.x < b.x; });
    float mx = v[m].x;
    std::nth_element(v.begin(), v.begin() + m, v.end(),
                     [](cv::Point2f a, cv::Point2f b){ return a.y < b.y; });
    float my = v[m].y;
    return {mx, my};
}

void trackFeatures(const std::vector<cv::Mat>& frames,
                   std::vector<TrackedPoint>& tracks) {
    tracks.clear();
    if (frames.size() < 2) return;

    // get good corners only on first frame.
    std::vector<cv::Point2f> p0;
    cv::goodFeaturesToTrack(frames[0], p0, kMaxCorners, kQualityLevel,
                            kMinDistance, cv::noArray(), kBlockSize);
    if (p0.empty()) return;

    tracks.reserve(p0.size());
    for (const auto& p : p0) {
        TrackedPoint t;
        t.first = t.last = p;
        t.trajectory.push_back(p);
        tracks.push_back(t);
    }

    // alive tracks for this frame.
    std::vector<int>          alive;
    std::vector<cv::Point2f>  prev_pts;
    alive.reserve(p0.size()); prev_pts.reserve(p0.size());
    for (size_t i = 0; i < p0.size(); ++i) {
        alive.push_back((int)i);
        prev_pts.push_back(p0[i]);
    }

    cv::TermCriteria term(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                          30, 0.01);

    for (size_t f = 1; f < frames.size(); ++f) {
        if (alive.empty()) break;

        std::vector<cv::Point2f> next_pts, back_pts;
        std::vector<uchar> ok_fwd, ok_bwd;
        std::vector<float> err_fwd, err_bwd;
        // forward: where points go in next frame.
        cv::calcOpticalFlowPyrLK(frames[f - 1], frames[f],
                                 prev_pts, next_pts, ok_fwd, err_fwd,
                                 {kPyrWinSize, kPyrWinSize}, kPyrMaxLevel,
                                 term);
        // backward: do we come back to same place? if not, kill track.
        cv::calcOpticalFlowPyrLK(frames[f], frames[f - 1],
                                 next_pts, back_pts, ok_bwd, err_bwd,
                                 {kPyrWinSize, kPyrWinSize}, kPyrMaxLevel,
                                 term);

        std::vector<int>         new_alive;
        std::vector<cv::Point2f> new_prev;
        std::vector<int>         frame_track_idx;
        std::vector<cv::Point2f> frame_disp;
        new_alive.reserve(alive.size());
        new_prev.reserve(alive.size());

        for (size_t a = 0; a < alive.size(); ++a) {
            if (!ok_fwd[a] || !ok_bwd[a]) continue;
            cv::Point2f drift = back_pts[a] - prev_pts[a];
            if (cv::norm(drift) > kForwardBackErr) continue;

            int ti = alive[a];
            cv::Point2f p = next_pts[a];
            tracks[ti].last = p;
            tracks[ti].trajectory.push_back(p);

            new_alive.push_back(ti);
            new_prev.push_back(p);
            frame_track_idx.push_back(ti);
            frame_disp.push_back(p - tracks[ti].first);
        }

        // remove camera motion = the median of all displacements.
        if (!frame_disp.empty()) {
            cv::Point2f cam = medianXY(frame_disp);
            for (size_t k = 0; k < frame_track_idx.size(); ++k) {
                int ti = frame_track_idx[k];
                cv::Point2f rel = frame_disp[k] - cam;
                tracks[ti].total_disp += (float)cv::norm(rel);
                tracks[ti].n_matches++;
            }
        }

        alive    = std::move(new_alive);
        prev_pts = std::move(new_prev);
    }
}

} // namespace dod
