#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct KTrack {
    int id;
    cv::KalmanFilter kf; // state: x,y,vx,vy
    cv::Rect bbox;
    int age;
    int totalVisibleCount;
    int consecutiveInvisibleCount;
};

class KalmanTracker {
public:
    KalmanTracker();
    int createTrack(const cv::Rect& bbox);
    void predict();
    void update(int trackId, const cv::Rect& bbox);
    void eraseLostTracks();
    std::vector<KTrack> getTracks() const;
private:
    int nextId;
    std::map<int, KTrack> tracks;
};
