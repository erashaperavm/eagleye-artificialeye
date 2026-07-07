#include "saliency.h"

SaliencyDetector::SaliencyDetector() {
}

void SaliencyDetector::update(const cv::Mat& frame) {
    // no-op fallback when OpenCV saliency is not available
    (void)frame;
}

void SaliencyDetector::draw(cv::Mat& img) {
    (void)img;
}
