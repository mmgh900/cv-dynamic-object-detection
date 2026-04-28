#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace dod {

// one tracked point. n_matches is how many frames it survived,
// total_disp is total movement after we remove camera motion.
struct TrackedPoint {
    cv::Point2f first;
    cv::Point2f last;
    std::vector<cv::Point2f> trajectory;
    int   n_matches  = 0;
    float total_disp = 0.f;
};

std::vector<std::string> listFrames(const std::string& folder);

void trackFeatures(const std::vector<cv::Mat>& frames_gray,
                   std::vector<TrackedPoint>& tracks);

std::vector<int> classifyDynamicTracks(const std::vector<TrackedPoint>& tracks);

cv::Rect estimateBBox(const std::vector<TrackedPoint>& tracks,
                      const std::vector<int>& dynamic_idx,
                      const cv::Size& img_size);

cv::Rect refineBBoxSilhouette(const std::vector<cv::Mat>& frames_gray,
                              const cv::Rect& coarse_box,
                              const cv::Size& img_size);

cv::Rect readGT(const std::string& path);
double   iou(const cv::Rect& a, const cv::Rect& b);
void     writePrediction(const std::string& path, const cv::Rect& box);

} // namespace dod
