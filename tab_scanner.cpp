#include "tab_scanner.h"
#include <opencv2/imgproc.hpp>
#include <thread>

#include "tab_scanner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <iostream>

TabScanner::TabScanner(const std::string& modelPath) : modelPath(modelPath) {
    // try to preload MiDaS model using OpenCV DNN (ONNX)
}

void TabScanner::scan(const cv::Mat& frame, std::function<void(const cv::Mat&)> cb) {
    if (frame.empty()) { if (cb) cb(cv::Mat()); return; }
    // PERFORMANCE: should run in separate thread, single-shot, target <=200ms
    if (!modelPath.empty()) {
        try {
            cv::dnn::Net net = cv::dnn::readNet(modelPath);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            cv::Mat input;
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            cv::resize(rgb, input, cv::Size(256,256));
            input.convertTo(input, CV_32F, 1.0/255.0);
            // normalize with mean/std if necessary
            cv::Mat blob = cv::dnn::blobFromImage(input);
            net.setInput(blob);
            cv::Mat out = net.forward();
            // out may be 1x1xHxW or 1xHxWx1 depending on model
            cv::Mat depth;
            if (out.dims == 4) {
                // shape N,C,H,W
                int H = out.size[2]; int W = out.size[3];
                depth = cv::Mat(H, W, CV_32F, out.ptr<float>());
            } else if (out.dims == 2) {
                depth = out.reshape(1, out.size[0]);
            } else {
                depth = out.clone();
            }
            cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX);
            depth.convertTo(depth, CV_8U);
            cv::Mat depthUp;
            cv::resize(depth, depthUp, frame.size());
            lastDepth = depthUp;
            if (cb) cb(depthUp);
            return;
        } catch (const std::exception &e) {
            std::cerr << "MiDaS inference failed: " << e.what() << "\n";
        }
    }

    // Fallback placeholder depth: use laplacian magnitude
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Mat absLap;
    cv::convertScaleAbs(lap, absLap);
    cv::Mat depth;
    cv::normalize(absLap, depth, 0, 255, cv::NORM_MINMAX);
    lastDepth = depth;
    if (cb) cb(depth);
}

void TabScanner::drawEffects(cv::Mat& img, bool depthReady) {
    // draw scanning lines
    int rows = img.rows;
    static int offset = 0;
    offset = (offset + 4) % rows;
    for (int y = 0; y < rows; y += 16) {
        int yy = (y + offset) % rows;
        cv::line(img, cv::Point(0, yy), cv::Point(img.cols, yy), cv::Scalar(0,255,255,50), 2);
    }
    // vignette
    cv::Mat mask(img.size(), CV_32F);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            float dx = (x - img.cols/2.0f)/(img.cols/2.0f);
            float dy = (y - img.rows/2.0f)/(img.rows/2.0f);
            float v = 1.0f - std::min(1.0f, sqrtf(dx*dx + dy*dy));
            mask.at<float>(y,x) = v;
        }
    }
    cv::Mat tmp;
    img.convertTo(tmp, CV_32F);
    std::vector<cv::Mat> ch; cv::split(tmp, ch);
    for (int i = 0; i < 3; ++i) ch[i] = ch[i].mul(mask);
    cv::merge(ch, tmp);
    tmp.convertTo(img, CV_8U);

    // color separation (simple channel shift)
    if (depthReady && !lastDepth.empty()) {
        cv::Mat colorDepth;
        cv::applyColorMap(lastDepth, colorDepth, cv::COLORMAP_JET);
        cv::addWeighted(img, 0.6, colorDepth, 0.4, 0, img);
    }
}
