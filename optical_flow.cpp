#include "optical_flow.h"
#include <opencv2/imgproc.hpp>

OpticalFlow::OpticalFlow() : paused(false) {}

void OpticalFlow::update(const cv::Mat& frame) {
    if (paused.load()) return;
    if (frame.empty()) return;
    // Compute on central 1/3 ROI, downsample to ~180p
    int w = frame.cols/3; int h = frame.rows/3; int x = frame.cols/3; int y = frame.rows/3;
    if (w <=0 || h <=0) return;
    cv::Rect roi(x,y,w,h);
    cv::Mat crop = frame(roi);
    if (crop.empty()) return;
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    if (!prevGray.empty()) {
        try { cv::calcOpticalFlowFarneback(prevGray, small, flow, 0.5, 3, 15, 3, 5, 1.2, 0); }
        catch(...) { flow = cv::Mat(); }
    }
    prevGray = small;
}

void OpticalFlow::computeOnFrame(const cv::Mat& frame) {
    // guard wrapper
    update(frame);
}

void OpticalFlow::draw(cv::Mat& img) {
    if (flow.empty()) return;
    // Draw sparse arrows every 16x16 in flow mat
    cv::Mat display = img;
    int step = 16;
    for (int y = 0; y < flow.rows; y += step) {
        for (int x = 0; x < flow.cols; x += step) {
            const cv::Point2f f = flow.at<cv::Point2f>(y, x);
            cv::Point2f p(static_cast<float>(x), static_cast<float>(y));
            cv::Point2f q = p + f * 5.0f;
            cv::line(display, p, q, cv::Scalar(255,0,0), 1);
            cv::circle(display, q, 1, cv::Scalar(255,0,0), -1);
        }
    }
}

void OpticalFlow::pause() { paused = true; }
void OpticalFlow::resume() { paused = false; }
