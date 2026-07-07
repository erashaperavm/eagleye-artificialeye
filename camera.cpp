#include "camera.h"
#include <iostream>

CameraCapture::CameraCapture(int index, int w, int h)
    : camIndex(index), width(w), height(h), zoomFactor(1.0) {
}
CameraCapture::~CameraCapture() { if (cap.isOpened()) cap.release(); }

bool CameraCapture::open() {
    cap.open(camIndex);
    if (!cap.isOpened()) return false;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    updateROI();
    return true;
}

void CameraCapture::updateROI() {
    int w = static_cast<int>(width / zoomFactor);
    int h = static_cast<int>(height / zoomFactor);
    int x = (width - w) / 2;
    int y = (height - h) / 2;
    roi = cv::Rect(x, y, w, h);
}

cv::Mat CameraCapture::grabFrame() {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return frame;
    // PERFORMANCE: avoid copies where possible; return ROI view
    cv::Mat cropped = frame(roi);
    return cropped; // caller may clone when needed
}

void CameraCapture::zoomIn() {
    zoomFactor = std::min(zoomFactor + 0.25, 4.0);
    updateROI();
}
void CameraCapture::zoomOut() {
    zoomFactor = std::max(zoomFactor - 0.25, 1.0);
    updateROI();
}
