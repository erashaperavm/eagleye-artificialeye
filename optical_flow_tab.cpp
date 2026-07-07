#include "optical_flow.h"

// Lightweight optical-flow-on-demand: compute flow only when provided a frame (Tab-mode)
void OpticalFlow::computeOnFrame(const cv::Mat& frame) {
    // reuse update logic but exposed as callable
    update(frame);
}
