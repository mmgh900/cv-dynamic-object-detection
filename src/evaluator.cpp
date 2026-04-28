// Author: Filippo Businaro
// read GT box and compute IoU.

#include "dod.hpp"
#include <fstream>

namespace dod {

cv::Rect readGT(const std::string& path) {
    std::ifstream in(path);
    int x0, y0, x1, y1;
    if (!(in >> x0 >> y0 >> x1 >> y1)) return {};
    return cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

double iou(const cv::Rect& a, const cv::Rect& b) {
    if (a.area() <= 0 || b.area() <= 0) return 0.0;
    double i = (a & b).area();
    double u = a.area() + b.area() - i;
    return u > 0 ? i / u : 0.0;
}

void writePrediction(const std::string& path, const cv::Rect& box) {
    std::ofstream out(path);
    out << box.x << " " << box.y << " "
        << (box.x + box.width) << " " << (box.y + box.height) << "\n";
}

} // namespace dod
