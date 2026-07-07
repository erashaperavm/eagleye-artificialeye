#pragma once
#include <opencv2/opencv.hpp>

class CameraCapture {
public:
    CameraCapture(int index = 0, int width = 1280, int height = 720);
    ~CameraCapture();
    bool open();
    cv::Mat grabFrame();
    void zoomIn();
    void zoomOut();
private:
    int camIndex;
    int width, height;
    cv::VideoCapture cap;
    double zoomFactor; // 1.0 - 4.0
    cv::Rect roi;
    void updateROI();
};
