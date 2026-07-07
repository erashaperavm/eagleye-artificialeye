#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>

class SaliencyDetector {
public:
    SaliencyDetector();
    void update(const cv::Mat& frame);
    void draw(cv::Mat& img);
private:
    cv::Mat salMap; // 0-255
};
