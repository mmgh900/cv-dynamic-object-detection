// run the pipeline on each category and save the outputs.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using namespace dod;

namespace {
constexpr int kMaxFrames = 80;
}

static void drawTracks(cv::Mat& img,
                       const std::vector<TrackedPoint>& tracks,
                       const std::vector<int>& dyn_idx) {
    std::vector<bool> is_dyn(tracks.size(), false);
    for (int i : dyn_idx) is_dyn[i] = true;
    for (size_t i = 0; i < tracks.size(); ++i) {
        cv::Scalar col = is_dyn[i] ? cv::Scalar(0, 0, 255)
                                   : cv::Scalar(0, 180, 255);
        cv::circle(img, tracks[i].first, 2, col, -1);
        if (is_dyn[i] && tracks[i].trajectory.size() > 1) {
            for (size_t k = 1; k < tracks[i].trajectory.size(); ++k)
                cv::line(img, tracks[i].trajectory[k - 1],
                         tracks[i].trajectory[k], cv::Scalar(0, 255, 0), 1);
        }
    }
}

struct RunOutput {
    std::string category;
    cv::Rect gt, pred;
    double iou = 0.0;
    int n_tracks = 0, n_dynamic = 0;
    cv::Size img_size;
};

static RunOutput runOne(const std::string& cat,
                        const std::string& data_dir,
                        const std::string& label_dir,
                        const std::string& out_dir) {
    RunOutput r{cat};
    auto frames = listFrames(data_dir);
    if (frames.empty()) {
        std::cerr << "no frames in " << data_dir << "\n";
        return r;
    }

    // Sub-sample long sequences so SIFT cost stays bounded.
    int stride = std::max(1, (int)frames.size() / kMaxFrames);
    cv::Mat first = cv::imread(frames[0]);
    r.img_size = first.size();

    std::vector<cv::Mat> gray;
    for (size_t i = 0; i < frames.size(); i += stride) {
        cv::Mat c = cv::imread(frames[i]);
        if (c.empty()) continue;
        cv::Mat g;
        cv::cvtColor(c, g, cv::COLOR_BGR2GRAY);
        gray.push_back(g);
        if ((int)gray.size() >= kMaxFrames) break;
    }

    std::vector<TrackedPoint> tracks;
    trackFeatures(gray, tracks);
    auto dyn      = classifyDynamicTracks(tracks);
    cv::Rect pred = estimateBBox(tracks, dyn, r.img_size);

    r.gt        = readGT(label_dir + "/0000.txt");
    r.pred      = pred;
    r.iou       = iou(r.gt, pred);
    r.n_tracks  = (int)tracks.size();
    r.n_dynamic = (int)dyn.size();

    writePrediction(out_dir + "/" + cat + "_pred.txt", pred);

    cv::Mat overlay = first.clone();
    drawTracks(overlay, tracks, dyn);
    if (r.gt.area()   > 0) cv::rectangle(overlay, r.gt,   {0, 255, 0}, 2);
    if (r.pred.area() > 0) cv::rectangle(overlay, r.pred, {0, 0, 255}, 2);
    std::ostringstream txt;
    txt << cat << "  IoU=" << std::fixed << std::setprecision(3) << r.iou;
    cv::putText(overlay, txt.str(), {10, 25},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
    cv::imwrite(out_dir + "/" + cat + "_overlay.png", overlay);

    cv::Mat clean = first.clone();
    if (r.gt.area()   > 0) cv::rectangle(clean, r.gt,   {0, 255, 0}, 2);
    if (r.pred.area() > 0) cv::rectangle(clean, r.pred, {0, 0, 255}, 2);
    cv::imwrite(out_dir + "/" + cat + "_boxes.png", clean);

    cv::Mat tv = first.clone();
    drawTracks(tv, tracks, dyn);
    cv::imwrite(out_dir + "/" + cat + "_tracks.png", tv);
    return r;
}

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : "dataset";
    std::string out  = (argc > 2) ? argv[2] : "output";
    fs::create_directories(out);

    const std::vector<std::string> cats =
        {"bird", "car", "frog", "sheep", "squirrel"};

    std::ofstream csv(out + "/results.csv");
    csv << "category,iou,correct,pred_x,pred_y,pred_w,pred_h,"
        << "gt_x,gt_y,gt_w,gt_h,n_tracks,n_dynamic\n";

    double sum_iou = 0.0;
    int correct = 0;
    for (const auto& c : cats) {
        std::cout << "[run] " << c << "\n";
        auto r = runOne(c, root + "/data/" + c, root + "/labels/" + c, out);
        sum_iou += r.iou;
        if (r.iou > 0.5) correct++;
        csv << c << "," << r.iou << "," << (r.iou > 0.5 ? 1 : 0) << ","
            << r.pred.x << "," << r.pred.y << ","
            << r.pred.width << "," << r.pred.height << ","
            << r.gt.x   << "," << r.gt.y   << ","
            << r.gt.width   << "," << r.gt.height   << ","
            << r.n_tracks << "," << r.n_dynamic << "\n";
        std::cout << "   IoU=" << r.iou
                  << " tracks=" << r.n_tracks
                  << " dyn="    << r.n_dynamic << "\n";
    }

    double mIoU = sum_iou / cats.size();
    double acc  = (double)correct / cats.size();
    std::ofstream sf(out + "/summary.txt");
    sf << "mIoU: " << mIoU << "\n"
       << "accuracy@0.5: " << acc << " (" << correct << "/" << cats.size() << ")\n";
    std::cout << "\nmIoU=" << mIoU << "  accuracy=" << acc << "\n";
    return 0;
}
