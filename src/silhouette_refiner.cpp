// Author: Filippo Businaro
// try to make the box smaller using pixel difference between frames.
// only take the new box if it is much smaller than the old one.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>

namespace dod {
namespace {
constexpr int    kKernelSize    = 5;
constexpr double kMinOverlap    = 0.3;
constexpr double kMaxAreaRatio  = 0.6;
constexpr int    kPadPx         = 6;
}

// how much did the camera move from a to b.
static cv::Point2f phaseShift(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat af, bf, hann;
    a.convertTo(af, CV_32F);
    b.convertTo(bf, CV_32F);
    cv::createHanningWindow(hann, a.size(), CV_32F);
    cv::Point2d s = cv::phaseCorrelate(af, bf, hann);
    return {(float)s.x, (float)s.y};
}

cv::Rect refineBBoxSilhouette(const std::vector<cv::Mat>& frames,
                              const cv::Rect& coarse,
                              const cv::Size& img_size) {
    if (frames.size() < 2 || coarse.area() <= 0) return coarse;

    const cv::Mat& ref = frames[0];
    cv::Mat accum = cv::Mat::zeros(img_size, CV_32F);
    int n = 0;
    // for each later frame: shift it back to frame 0, then take the
    // difference. moving pixels light up.
    for (size_t f = 1; f < frames.size(); ++f) {
        cv::Point2f s = phaseShift(ref, frames[f]);
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -s.x,
                                               0, 1, -s.y);
        cv::Mat warped, diff, diff_f;
        cv::warpAffine(frames[f], warped, M, img_size,
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::absdiff(ref, warped, diff);
        diff.convertTo(diff_f, CV_32F);
        accum += diff_f;
        n++;
    }
    if (n == 0) return coarse;
    accum /= (float)n;

    // make a mask: bright = moving, dark = still.
    cv::Mat mean8, mask;
    accum.convertTo(mean8, CV_8U);
    cv::GaussianBlur(mean8, mean8, {kKernelSize, kKernelSize}, 0);
    cv::threshold(mean8, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                          {kKernelSize, kKernelSize});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    // find the biggest blob that touches the coarse box.
    cv::Mat labels, stats, cents;
    int nc = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
    if (nc <= 1) return coarse;

    int best = -1, best_area = 0;
    for (int i = 1; i < nc; ++i) {
        cv::Rect cc(stats.at<int>(i, cv::CC_STAT_LEFT),
                    stats.at<int>(i, cv::CC_STAT_TOP),
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT));
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if ((cc & coarse).area() <= 0) continue;
        if (a > best_area) { best_area = a; best = i; }
    }
    if (best < 0) return coarse;

    cv::Rect refined(stats.at<int>(best, cv::CC_STAT_LEFT),
                     stats.at<int>(best, cv::CC_STAT_TOP),
                     stats.at<int>(best, cv::CC_STAT_WIDTH),
                     stats.at<int>(best, cv::CC_STAT_HEIGHT));

    // only accept if it overlaps and is much smaller.
    double ov = (double)(refined & coarse).area() /
                std::max(1, std::min(refined.area(), coarse.area()));
    if (ov < kMinOverlap) return coarse;
    if (refined.area() > kMaxAreaRatio * coarse.area()) return coarse;

    refined.x      -= kPadPx;
    refined.y      -= kPadPx;
    refined.width  += 2 * kPadPx;
    refined.height += 2 * kPadPx;
    return refined & cv::Rect(0, 0, img_size.width, img_size.height);
}

} // namespace dod
