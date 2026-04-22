// Author: Mahdi Gheysari
// Module owner: Mahdi Gheysari (feature detection, matching, median-flow).
//
// SIFT feature extraction + nearest-neighbour descriptor matching with
// Lowe's ratio test. Produces, for every keypoint detected in the first
// frame, its tracked position in subsequent frames by chaining matches
// between frame 0 and every later frame.

#include "dod.hpp"
#include <opencv2/features2d.hpp>
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

// Kept for interface compatibility; no longer used by trackFeatures,
// but other callers (e.g. diagnostics) can still detect corners.
std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray_img, int max_corners) {
    auto sift = cv::SIFT::create(max_corners);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    sift->detectAndCompute(gray_img, cv::noArray(), kps, desc);
    std::vector<cv::Point2f> out;
    out.reserve(kps.size());
    for (const auto& k : kps) out.push_back(k.pt);
    return out;
}

// For every SIFT keypoint detected in frame 0, match it against SIFT
// keypoints in every subsequent frame using BFMatcher L2 knnMatch(k=2)
// filtered by Lowe's ratio test at 0.75. Each successful match records
// the matched position along the track.
void trackFeatures(const std::vector<cv::Mat>& frames_gray,
                   std::vector<TrackedPoint>& tracks) {
    tracks.clear();
    if (frames_gray.size() < 2) return;

    const int n_features = 1500;
    const float lowe_ratio = 0.75f;

    auto sift = cv::SIFT::create(n_features);

    std::vector<cv::KeyPoint> kp0;
    cv::Mat desc0;
    sift->detectAndCompute(frames_gray[0], cv::noArray(), kp0, desc0);

    tracks.reserve(kp0.size());
    for (const auto& k : kp0) {
        TrackedPoint t;
        t.first = k.pt;
        t.last  = k.pt;
        t.trajectory.push_back(k.pt);
        t.inlier_count  = 0;  // repurposed: count of successful matches
        t.outlier_count = 0;  // repurposed: count of observations above bg scale
        t.residual      = 0.f; // repurposed: total displacement from frame 0
        tracks.push_back(t);
    }

    cv::BFMatcher matcher(cv::NORM_L2, /*crossCheck=*/false);

    for (size_t f = 1; f < frames_gray.size(); ++f) {
        std::vector<cv::KeyPoint> kpF;
        cv::Mat descF;
        sift->detectAndCompute(frames_gray[f], cv::noArray(), kpF, descF);
        if (descF.empty() || desc0.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc0, descF, knn, 2);

        // First pass: collect valid matches for this frame so we can
        // estimate a global translation (median flow) and remove it,
        // which covers the slight camera pan in the car sequence
        // without fitting a full homography.
        struct Mtch { int i; cv::Point2f p_new; cv::Point2f disp; };
        std::vector<Mtch> ok;
        ok.reserve(knn.size());
        for (const auto& pair : knn) {
            if (pair.size() < 2) continue;
            if (pair[0].distance >= lowe_ratio * pair[1].distance) continue;
            int i = pair[0].queryIdx;
            int j = pair[0].trainIdx;
            if (i < 0 || i >= (int)tracks.size()) continue;
            if (j < 0 || j >= (int)kpF.size())    continue;
            cv::Point2f p_new = kpF[j].pt;
            ok.push_back({i, p_new, p_new - tracks[i].first});
        }
        if (ok.empty()) continue;

        // Median displacement across matched keypoints.
        std::vector<float> xs, ys;
        xs.reserve(ok.size()); ys.reserve(ok.size());
        for (const auto& m : ok) { xs.push_back(m.disp.x); ys.push_back(m.disp.y); }
        std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
        std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
        cv::Point2f med(xs[xs.size() / 2], ys[ys.size() / 2]);

        for (const auto& m : ok) {
            tracks[m.i].last = m.p_new;
            tracks[m.i].trajectory.push_back(m.p_new);
            cv::Point2f rel = m.disp - med;   // motion relative to global flow
            tracks[m.i].residual += (float)cv::norm(rel);
            tracks[m.i].inlier_count++;
        }
    }
}

} // namespace dod
