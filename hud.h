#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>

class HUDManager {
public:
    HUDManager();
    void pushMessage(const std::string& s);
    void draw(cv::Mat& img);
private:
    std::deque<std::string> msgs;
    std::mutex mtx;
};
