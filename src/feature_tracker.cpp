// Author: Mahdi Gheysari
// Sparse feature detection and KLT tracking across an image sequence.

#include "dod.hpp"
#include <opencv2/video/tracking.hpp>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace dod {

std::vector<std::string> listFrames(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray_img, int max_corners) {
    std::vector<cv::Point2f> corners;
    // Adaptive min distance: smaller images need denser sampling so
    // that small moving objects (squirrel, car) get enough features.
    double diag = std::sqrt((double)(gray_img.cols * gray_img.cols +
                                     gray_img.rows * gray_img.rows));
    double min_dist = std::max(3.0, diag / 180.0);
    cv::goodFeaturesToTrack(gray_img, corners,
                            max_corners,
                            0.003,   // quality level (low -> dense)
                            min_dist,
                            cv::noArray(),
                            5,       // block size
                            false,   // use Harris
                            0.04);
    if (!corners.empty()) {
        cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03);
        cv::cornerSubPix(gray_img, corners, cv::Size(7, 7), cv::Size(-1, -1), crit);
    }
    return corners;
}

void trackFeatures(const std::vector<cv::Mat>& frames_gray,
                   std::vector<TrackedPoint>& tracks) {
    if (frames_gray.size() < 2) return;

    auto seeds = detectFeatures(frames_gray[0]);
    tracks.clear();
    tracks.reserve(seeds.size());
    for (const auto& p : seeds) {
        TrackedPoint t;
        t.first = p;
        t.last = p;
        t.trajectory.push_back(p);
        tracks.push_back(t);
    }

    const int win = 21;
    cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);

    std::vector<cv::Point2f> prev_pts;
    for (auto& t : tracks) prev_pts.push_back(t.last);

    for (size_t f = 1; f < frames_gray.size(); ++f) {
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> status;
        std::vector<float> err;

        if (prev_pts.empty()) break;

        cv::calcOpticalFlowPyrLK(frames_gray[f - 1], frames_gray[f],
                                 prev_pts, next_pts, status, err,
                                 cv::Size(win, win), 3, crit);

        // Backward check for robustness
        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> status_b;
        std::vector<float> err_b;
        cv::calcOpticalFlowPyrLK(frames_gray[f], frames_gray[f - 1],
                                 next_pts, back_pts, status_b, err_b,
                                 cv::Size(win, win), 3, crit);

        size_t k = 0;
        for (auto& t : tracks) {
            if (!t.alive) continue;
            if (k >= status.size()) break;
            bool ok = status[k] && status_b[k];
            if (ok) {
                float dx = back_pts[k].x - prev_pts[k].x;
                float dy = back_pts[k].y - prev_pts[k].y;
                if (dx * dx + dy * dy > 1.5f) ok = false;
            }
            if (!ok) {
                t.alive = false;
            } else {
                t.last = next_pts[k];
                t.trajectory.push_back(next_pts[k]);
            }
            ++k;
        }

        // Rebuild prev_pts from alive tracks
        prev_pts.clear();
        for (auto& t : tracks) if (t.alive) prev_pts.push_back(t.last);
    }
}

} // namespace dod
