#pragma once
#include <opencv2/opencv.hpp>
#include <functional>
#include <string>

class TabScanner {
public:
    TabScanner(const std::string& modelPath);
    void scan(const cv::Mat& frame, std::function<void(const cv::Mat&)> cb);
    void drawEffects(cv::Mat& img, bool depthReady);
private:
    std::string modelPath;
    cv::Mat lastDepth;
};
