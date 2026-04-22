// Author: Filippo Businaro
// Module owner: Filippo Businaro (ground-truth I/O and IoU metrics).

#include "dod.hpp"
#include <fstream>
#include <sstream>

namespace dod {

cv::Rect readGT(const std::string& path) {
    std::ifstream in(path);
    int xmin, ymin, xmax, ymax;
    if (!(in >> xmin >> ymin >> xmax >> ymax)) return cv::Rect();
    return cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
}

double iou(const cv::Rect& a, const cv::Rect& b) {
    if (a.area() <= 0 || b.area() <= 0) return 0.0;
    cv::Rect inter = a & b;
    double i = inter.area();
    double u = a.area() + b.area() - i;
    return u > 0 ? i / u : 0.0;
}

void writePrediction(const std::string& path, const cv::Rect& box) {
    std::ofstream out(path);
    out << box.x << " " << box.y << " "
        << (box.x + box.width) << " " << (box.y + box.height) << "\n";
}

} // namespace dod
