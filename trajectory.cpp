#include "trajectory.h"
#include "tracker.h"

TrajectoryPredictor::TrajectoryPredictor() {}

void TrajectoryPredictor::update(const std::vector<Track>& tracks) {
    for (const auto& t : tracks) {
        auto center = cv::Point2f(t.bbox.x + t.bbox.width/2.0f, t.bbox.y + t.bbox.height/2.0f);
        auto &vec = history[t.id];
        vec.push_back(center);
        if (vec.size() > 10) vec.erase(vec.begin());
    }
}

void TrajectoryPredictor::draw(cv::Mat& img, const std::vector<int>& focusIds) {
    for (const auto &kv : history) {
        int id = kv.first;
        if (!focusIds.empty() && std::find(focusIds.begin(), focusIds.end(), id) == focusIds.end()) continue;
        const auto &vec = kv.second;
        for (size_t i = 1; i < vec.size(); ++i) {
            cv::line(img, vec[i-1], vec[i], cv::Scalar(0,255,255), 2);
        }
        // predict uniform velocity for 0.5s (mock)
        if (vec.size() >= 2) {
            auto a = vec[vec.size()-2];
            auto b = vec.back();
            cv::Point2f v = (b - a);
            cv::Point2f pred = b + v * 12.0f; // simplistic
            cv::circle(img, pred, 4, cv::Scalar(0,128,255), -1);
        }
    }
}
