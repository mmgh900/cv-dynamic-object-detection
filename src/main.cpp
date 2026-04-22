// Author: Mahdi Gheysari
// Module owner: Mahdi Gheysari (CLI entry, per-category orchestration,
// visualisation overlays).
//
// Runs the full pipeline on every dataset category and writes
// predictions, overlay images, and an evaluation report.

#include "dod.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace dod;

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
            for (size_t k = 1; k < tracks[i].trajectory.size(); ++k) {
                cv::line(img, tracks[i].trajectory[k - 1],
                         tracks[i].trajectory[k], cv::Scalar(0, 255, 0), 1);
            }
        }
    }
}

struct RunOutput {
    std::string category;
    cv::Rect gt, pred;
    double iou;
    int n_tracks, n_dynamic;
    cv::Size img_size;
};

static RunOutput processCategory(const std::string& category,
                                 const std::string& data_dir,
                                 const std::string& label_dir,
                                 const std::string& out_dir) {
    RunOutput r;
    r.category = category;
    auto frames = listFrames(data_dir);
    if (frames.empty()) { std::cerr << "no frames in " << data_dir << "\n"; return r; }

    // Load gray frames. Use a stride to bound work on long sequences.
    std::vector<cv::Mat> gray;
    const int max_frames = 80;
    int stride = 1;
    if ((int)frames.size() > max_frames)
        stride = std::max(1, (int)frames.size() / max_frames);
    cv::Mat first_color = cv::imread(frames[0]);
    r.img_size = first_color.size();
    for (size_t i = 0; i < frames.size(); i += stride) {
        cv::Mat c = cv::imread(frames[i]);
        if (c.empty()) continue;
        cv::Mat g; cv::cvtColor(c, g, cv::COLOR_BGR2GRAY);
        gray.push_back(g);
        if ((int)gray.size() >= max_frames) break;
    }

    std::vector<TrackedPoint> tracks;
    trackFeatures(gray, tracks);
    auto dyn = classifyDynamicTracks(gray, tracks);
    cv::Rect coarse = estimateBBox(tracks, dyn, r.img_size);
    cv::Rect pred   = refineBBoxSilhouette(gray, coarse, r.img_size);
    auto all_clusters = clusterBBoxes(tracks, dyn, r.img_size);

    cv::Rect gt = readGT(label_dir + "/0000.txt");
    r.gt = gt; r.pred = pred;
    r.iou = iou(gt, pred);
    r.n_tracks = (int)tracks.size();
    r.n_dynamic = (int)dyn.size();

    // Write prediction txt.
    writePrediction(out_dir + "/" + category + "_pred.txt", pred);

    // Overlay image: GT green, prediction red, tracks orange/red.
    cv::Mat overlay = first_color.clone();
    drawTracks(overlay, tracks, dyn);
    // Thin yellow boxes: every motion cluster the system found. Helps
    // show the "multiple moving sub-objects merged into one" behaviour
    // required by the spec (e.g. two squirrels, multiple sheep).
    for (const auto& cb : all_clusters)
        cv::rectangle(overlay, cb, cv::Scalar(0, 255, 255), 1);
    if (gt.area() > 0) cv::rectangle(overlay, gt, cv::Scalar(0, 255, 0), 2);
    if (pred.area() > 0) cv::rectangle(overlay, pred, cv::Scalar(0, 0, 255), 2);

    std::cout << "   clusters=" << all_clusters.size();
    for (const auto& cb : all_clusters)
        std::cout << " [" << cb.x << "," << cb.y
                  << "," << cb.width << "x" << cb.height << "]";
    std::cout << "\n";

    std::stringstream txt;
    txt << category << "  IoU=" << std::fixed << std::setprecision(3) << r.iou;
    cv::putText(overlay, txt.str(), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::imwrite(out_dir + "/" + category + "_overlay.png", overlay);

    // Clean overlay (boxes only) for the report.
    cv::Mat clean = first_color.clone();
    if (gt.area() > 0) cv::rectangle(clean, gt, cv::Scalar(0, 255, 0), 2);
    if (pred.area() > 0) cv::rectangle(clean, pred, cv::Scalar(0, 0, 255), 2);
    cv::imwrite(out_dir + "/" + category + "_boxes.png", clean);

    // Tracks-only view (used as a supplementary figure).
    cv::Mat trackview = first_color.clone();
    drawTracks(trackview, tracks, dyn);
    cv::imwrite(out_dir + "/" + category + "_tracks.png", trackview);

    return r;
}

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : "dataset";
    std::string out  = (argc > 2) ? argv[2] : "output";
    fs::create_directories(out);

    std::vector<std::string> cats = {"bird", "car", "frog", "sheep", "squirrel"};
    std::vector<RunOutput> results;
    double sum_iou = 0.0; int correct = 0;

    std::ofstream csv(out + "/results.csv");
    csv << "category,iou,correct,pred_x,pred_y,pred_w,pred_h,"
        << "gt_x,gt_y,gt_w,gt_h,n_tracks,n_dynamic\n";

    for (const auto& c : cats) {
        std::cout << "[run] " << c << "\n";
        auto r = processCategory(c,
                                 root + "/data/"   + c,
                                 root + "/labels/" + c,
                                 out);
        results.push_back(r);
        sum_iou += r.iou;
        if (r.iou > 0.5) correct++;
        csv << c << "," << r.iou << "," << (r.iou > 0.5 ? 1 : 0) << ","
            << r.pred.x << "," << r.pred.y << "," << r.pred.width << "," << r.pred.height << ","
            << r.gt.x   << "," << r.gt.y   << "," << r.gt.width   << "," << r.gt.height   << ","
            << r.n_tracks << "," << r.n_dynamic << "\n";
        std::cout << "   IoU=" << r.iou
                  << " tracks=" << r.n_tracks
                  << " dyn="    << r.n_dynamic << "\n";
    }

    double mIoU = sum_iou / results.size();
    double acc  = double(correct) / results.size();
    std::ofstream sum(out + "/summary.txt");
    sum << "mIoU: " << mIoU << "\n"
        << "accuracy@0.5: " << acc << " (" << correct << "/" << results.size() << ")\n";
    std::cout << "\nmIoU=" << mIoU << "  accuracy=" << acc << "\n";
    return 0;
}
