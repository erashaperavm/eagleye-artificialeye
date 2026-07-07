#include "kalman_tracker.h"
#include <cmath>

KalmanTracker::KalmanTracker() : nextId(1) {}

int KalmanTracker::createTrack(const cv::Rect& bbox) {
    KTrack kt;
    kt.id = nextId++;
    // 4 state: x, y, vx, vy
    kt.kf.init(4, 2, 0);
    kt.kf.transitionMatrix = (cv::Mat_<float>(4,4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
    cv::setIdentity(kt.kf.measurementMatrix);
    cv::setIdentity(kt.kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kt.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kt.kf.errorCovPost, cv::Scalar::all(1));
    float cx = bbox.x + bbox.width/2.0f;
    float cy = bbox.y + bbox.height/2.0f;
    kt.kf.statePost.at<float>(0) = cx;
    kt.kf.statePost.at<float>(1) = cy;
    kt.kf.statePost.at<float>(2) = 0;
    kt.kf.statePost.at<float>(3) = 0;
    kt.bbox = bbox;
    kt.age = 1;
    kt.totalVisibleCount = 1;
    kt.consecutiveInvisibleCount = 0;
    tracks[kt.id] = kt;
    return kt.id;
}

void KalmanTracker::predict() {
    for (auto &kv : tracks) {
        auto &kt = kv.second;
        cv::Mat pred = kt.kf.predict();
        float px = pred.at<float>(0);
        float py = pred.at<float>(1);
        // keep bbox center at predicted
        int w = kt.bbox.width;
        int h = kt.bbox.height;
        kt.bbox.x = static_cast<int>(px - w/2.0f);
        kt.bbox.y = static_cast<int>(py - h/2.0f);
        kt.age++;
        kt.consecutiveInvisibleCount++;
    }
}

void KalmanTracker::update(int trackId, const cv::Rect& bbox) {
    auto it = tracks.find(trackId);
    if (it == tracks.end()) return;
    auto &kt = it->second;
    float cx = bbox.x + bbox.width/2.0f;
    float cy = bbox.y + bbox.height/2.0f;
    cv::Mat meas = (cv::Mat_<float>(2,1) << cx, cy);
    kt.kf.correct(meas);
    kt.bbox = bbox;
    kt.totalVisibleCount++;
    kt.consecutiveInvisibleCount = 0;
}

void KalmanTracker::eraseLostTracks() {
    std::vector<int> toErase;
    for (const auto &kv : tracks) {
        const auto &kt = kv.second;
        if (kt.consecutiveInvisibleCount > 10) toErase.push_back(kv.first);
    }
    for (int id : toErase) tracks.erase(id);
}

std::vector<KTrack> KalmanTracker::getTracks() const {
    std::vector<KTrack> out;
    for (const auto &kv : tracks) out.push_back(kv.second);
    return out;
}
