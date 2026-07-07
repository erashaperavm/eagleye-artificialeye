#include "saliency.h"
#include <opencv2/saliency.hpp>

SaliencyDetector::SaliencyDetector() {}

void SaliencyDetector::update(const cv::Mat& frame) {
    if (frame.empty()) { salMap = cv::Mat(); return; }
    // PERFORMANCE: use Spectral Residual (fast)
    try {
#ifdef HAVE_OPENCV_SALIENCY
        cv::Ptr<cv::saliency::StaticSaliencySpectralResidual> sal = cv::saliency::StaticSaliencySpectralResidual::create();
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat salRes;
        sal->computeSaliency(frame, salRes);
        salRes.convertTo(salMap, CV_8U, 255.0);
#else
        salMap = cv::Mat::zeros(frame.size(), CV_8U);
#endif
    } catch (...) {
        salMap = cv::Mat::zeros(frame.size(), CV_8U);
    }
}

void SaliencyDetector::draw(cv::Mat& img) {
    if (salMap.empty()) return;
    cv::Mat heat;
    cv::applyColorMap(salMap, heat, cv::COLORMAP_JET);
    cv::addWeighted(img, 0.7, heat, 0.3, 0, img);
}
