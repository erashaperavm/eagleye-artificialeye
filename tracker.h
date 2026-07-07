#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <atomic>
#include <vector>
#include <map>

#include "kalman_tracker.h"

struct Track {
    int id;
    cv::Rect bbox;
    float confidence;
    cv::Point2f velocity; // pix/frame
    int missed;
};

class ObjectTracker {
public:
    ObjectTracker(const std::string& modelPath);
    void update(const cv::Mat& frame);
    void draw(cv::Mat& img, const std::vector<int>& selectedIds = {});
    void pause();
    void resume();
    std::vector<Track> getTracks();
    void setCameraParams(float focal_px, float person_height_m);
    int pickTargetAt(const cv::Point& p) const; // returns track id or -1
private:
    std::string modelPath;
    std::atomic<bool> paused;
    std::vector<Track> tracks;
    int nextId;
    // DNN model (YOLOv8 ONNX or OpenVINO IR via OpenCV-DNN)
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;
    // Kalman-based tracker instance
    KalmanTracker ktracker;
    // background subtractor fallback
    cv::Ptr<cv::BackgroundSubtractor> bgSub;
    // camera params for distance estimation
    float focal_px;
    float person_height_m;
    // PERFORMANCE: preallocated buffers could go here
    void detect(const cv::Mat& frame, std::vector<cv::Rect>& boxes, std::vector<float>& scores);
    static float iou(const cv::Rect& a, const cv::Rect& b);
};
