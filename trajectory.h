#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class TrajectoryPredictor {
public:
    TrajectoryPredictor();
    void update(const std::vector<struct Track>& tracks);
    void draw(cv::Mat& img, const std::vector<int>& focusIds = {});
private:
    // store last positions by id
    std::map<int, std::vector<cv::Point2f>> history;
};
