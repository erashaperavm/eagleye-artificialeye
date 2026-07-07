#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <vector>
#include <filesystem>

class PoseEstimator {
public:
    PoseEstimator(const std::string& modelPath);
    void update(const cv::Mat& frame);
    void draw(cv::Mat& img);
    void pause();
    void resume();
private:
    std::string modelPath;
    std::atomic<bool> paused;
    std::vector<cv::Point2f> keypoints; // pixel coords
    cv::dnn::Net net;
    bool usable;
};
