#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>

class OpticalFlow {
public:
    OpticalFlow();
    void update(const cv::Mat& frame);
    void computeOnFrame(const cv::Mat& frame); // exposed for Tab-mode
    void draw(cv::Mat& img);
    void pause();
    void resume();
private:
    std::atomic<bool> paused;
    cv::Mat prevGray;
    cv::Mat flow; // CV_32FC2
};
