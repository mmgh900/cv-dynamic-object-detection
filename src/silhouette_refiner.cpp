// Author: Filippo Businaro
// Module owner: Filippo Businaro (silhouette-based bbox refinement via
// frame differencing, morphology, and connected components).
//
// The feature-based pipeline (SIFT + Lowe + displacement) tends to
// include a few false-positive "dynamic" keypoints on the background,
// which after clustering + padding produces a bounding box that is
// systematically larger than the ground truth. This module takes the
// coarse bbox and refines it using a pixel-level motion signal:
//
//   (a) estimate per-frame camera translation with phase correlation
//       so the method also works on the slightly-panning car sequence;
//   (b) warp each later frame back to frame-0 coordinates and
//       accumulate |I_f - I_0|;
//   (c) Otsu-threshold + morphological open/close to get a clean
//       motion mask;
//   (d) take the largest connected component that overlaps the
//       coarse bbox and return its bounding rectangle.
//
// If the refined box looks implausible (no overlap with the coarse
// box, or the CC is weirdly large) we fall back to the coarse box.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>

namespace dod {

// Estimate a pure-translation camera shift from frame 0 to frame f
// using phase correlation on a Hanning-windowed gray image pair.
// Phase correlation tolerates a small moving foreground because the
// background provides the dominant Fourier-peak.
static cv::Point2f estimateTranslation(const cv::Mat& ref_gray,
                                       const cv::Mat& cur_gray) {
    cv::Mat ref_f, cur_f, hann;
    ref_gray.convertTo(ref_f, CV_32F);
    cur_gray.convertTo(cur_f, CV_32F);
    cv::createHanningWindow(hann, ref_gray.size(), CV_32F);
    cv::Point2d shift = cv::phaseCorrelate(ref_f, cur_f, hann);
    return cv::Point2f((float)shift.x, (float)shift.y);
}

cv::Rect refineBBoxSilhouette(const std::vector<cv::Mat>& frames_gray,
                              const cv::Rect& coarse_box,
                              const cv::Size& img_size) {
    if (frames_gray.size() < 2 || coarse_box.area() <= 0) return coarse_box;

    const cv::Mat& ref = frames_gray[0];

    // Accumulate the mean absolute difference across warped frames.
    cv::Mat accum = cv::Mat::zeros(img_size, CV_32F);
    int n_used = 0;
    for (size_t f = 1; f < frames_gray.size(); ++f) {
        cv::Point2f shift = estimateTranslation(ref, frames_gray[f]);

        // Warp frame f back onto frame 0 coordinates (undo the shift).
        cv::Mat M = (cv::Mat_<double>(2, 3) <<
                     1.0, 0.0, -shift.x,
                     0.0, 1.0, -shift.y);
        cv::Mat warped;
        cv::warpAffine(frames_gray[f], warped, M, img_size,
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat diff;
        cv::absdiff(ref, warped, diff);
        cv::Mat diff_f;
        diff.convertTo(diff_f, CV_32F);
        accum += diff_f;
        n_used++;
    }
    if (n_used == 0) return coarse_box;
    accum /= (float)n_used;

    // Smooth and threshold. Otsu on the 8-bit version auto-picks the
    // split between "background noise" and "object motion".
    cv::Mat mean8;
    accum.convertTo(mean8, CV_8U);
    cv::GaussianBlur(mean8, mean8, cv::Size(5, 5), 0);
    cv::Mat mask;
    cv::threshold(mean8, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Morphological clean-up: open to kill salt-and-pepper noise,
    // close to fill small holes inside the object.
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    // Connected components; pick the largest that overlaps coarse_box.
    cv::Mat labels, stats, cents;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
    if (n <= 1) return coarse_box; // only background

    int best = -1;
    int best_area = 0;
    for (int i = 1; i < n; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Rect cc_box(x, y, w, h);
        if ((cc_box & coarse_box).area() <= 0) continue;
        if (a > best_area) { best_area = a; best = i; }
    }
    if (best < 0) return coarse_box;

    int x = stats.at<int>(best, cv::CC_STAT_LEFT);
    int y = stats.at<int>(best, cv::CC_STAT_TOP);
    int w = stats.at<int>(best, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(best, cv::CC_STAT_HEIGHT);
    cv::Rect refined(x, y, w, h);

    // Sanity checks: the refined box must significantly overlap the
    // coarse box. If it doesn't (e.g. warp failed on a long parallax
    // sequence), keep coarse.
    double overlap = (double)(refined & coarse_box).area() /
                     (double)std::max(1, std::min(refined.area(), coarse_box.area()));
    if (overlap < 0.3) return coarse_box;

    // Only trust the refined box when it is *significantly* smaller
    // than the coarse one. When they have similar size, the coarse
    // bbox is already well-calibrated by the keypoint cluster, and
    // switching to the silhouette hurts because absdiff only fires on
    // high-contrast edges and misses smooth interior regions (e.g. the
    // frog's back, the car body). Empirically, a refined/coarse area
    // ratio below 0.6 indicates that coarse was inflated by false
    // positive keypoints and refinement is the right move.
    if ((double)refined.area() > 0.6 * (double)coarse_box.area())
        return coarse_box;

    // Small padding (6 px) to recover edges lost by erosion and to
    // re-include any silhouette pixels missed by the absdiff mask.
    const int pad = 6;
    refined.x      -= pad;
    refined.y      -= pad;
    refined.width  += 2 * pad;
    refined.height += 2 * pad;
    return refined & cv::Rect(0, 0, img_size.width, img_size.height);
}

} // namespace dod
