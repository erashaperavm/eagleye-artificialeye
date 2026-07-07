#include "pose.h"
#include <iostream>
#include <opencv2/dnn.hpp>

PoseEstimator::PoseEstimator(const std::string& modelPath)
    : modelPath(modelPath), paused(false), usable(false) {
    // try load lightweight pose model via OpenCV DNN if provided
    try {
        if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
            net = cv::dnn::readNet(modelPath);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            usable = true;
        }
    } catch (const std::exception &e) {
        std::cerr << "Warning: failed to load pose model: " << e.what() << "\n";
        usable = false;
    }
}

void PoseEstimator::update(const cv::Mat& frame) {
    if (paused.load()) return;
    keypoints.clear();
    if (frame.empty()) return; // guard
    if (!usable) return;
    try {
        cv::Mat input;
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, input, cv::Size(256,256));
        input.convertTo(input, CV_32F, 1.0/255.0);
        cv::Mat blob = cv::dnn::blobFromImage(input);
        net.setInput(blob);
        cv::Mat out = net.forward();
        // MoveNet ONNX often contains control-flow not supported by OpenCV; if forward succeeds,
        // parse output for typical MoveNet shape [1,1,17,3] or [1,17,3]
        if (out.dims >= 2) {
            std::vector<int> dims(out.size.p, out.size.p + out.dims);
            // try to find last two dims as keypoints
            if (out.total() >= 17*3) {
                const float* data = out.ptr<float>();
                int kp_count = (int)(out.total() / 3);
                for (int i = 0; i < kp_count; ++i) {
                    float y = data[i*3+0];
                    float x = data[i*3+1];
                    // score = data[i*3+2]
                    cv::Point2f p(x * frame.cols, y * frame.rows);
                    keypoints.push_back(p);
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Pose model inference failed: " << e.what() << "\n";
        usable = false;
        return;
    }
}

void PoseEstimator::draw(cv::Mat& img) {
    for (const auto& p : keypoints) {
        cv::circle(img, p, 3, cv::Scalar(0,0,255), -1);
    }
}

void PoseEstimator::pause() { paused = true; }
void PoseEstimator::resume() { paused = false; }
