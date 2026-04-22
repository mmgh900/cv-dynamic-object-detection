// Module owner: Riccardo Pesce (shared API — TrackedPoint struct and
// cross-module function declarations).
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace dod {

struct TrackedPoint {
    cv::Point2f first;
    cv::Point2f last;
    std::vector<cv::Point2f> trajectory;
    bool alive = true;
    float residual = 0.f;
    int inlier_count = 0;
    int outlier_count = 0;
};

// feature_tracker.cpp
std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray_img, int max_corners = 1500);
void trackFeatures(const std::vector<cv::Mat>& frames_gray,
                   std::vector<TrackedPoint>& tracks);

// motion_segmenter.cpp
// Returns index list of tracks considered dynamic. Implementation uses
// a displacement-percentile criterion; the third argument is kept for
// API compatibility and is unused.
std::vector<int> classifyDynamicTracks(const std::vector<cv::Mat>& frames_gray,
                                       std::vector<TrackedPoint>& tracks,
                                       double unused = 0.0);

// bbox_estimator.cpp
cv::Rect estimateBBox(const std::vector<TrackedPoint>& tracks,
                      const std::vector<int>& dynamic_idx,
                      const cv::Size& img_size);

// evaluator.cpp
struct EvalResult {
    std::string category;
    cv::Rect gt;
    cv::Rect pred;
    double iou;
    bool correct; // IoU > 0.5
};

cv::Rect readGT(const std::string& path);
double iou(const cv::Rect& a, const cv::Rect& b);
void writePrediction(const std::string& path, const cv::Rect& box);

// frame loading util
std::vector<std::string> listFrames(const std::string& folder);

} // namespace dod
