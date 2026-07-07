#include "tracker.h"
#include <iostream>

ObjectTracker::ObjectTracker(const std::string& modelPath)
    : modelPath(modelPath), paused(false), nextId(1), confThreshold(0.5f), nmsThreshold(0.4f), focal_px(800.0f), person_height_m(1.7f) {
    try {
        if (!modelPath.empty()) {
            net = cv::dnn::readNet(modelPath);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } catch (...) {
        std::cerr << "Warning: failed to load detector model: " << modelPath << "\n";
    }
    // initialize background subtractor for fallback
    try { bgSub = cv::createBackgroundSubtractorMOG2(); } catch(...) { bgSub = nullptr; }
}

float ObjectTracker::iou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    int area = a.area() + b.area() - inter;
    return area > 0 ? (float)inter / (float)area : 0.f;
}

void ObjectTracker::detect(const cv::Mat& frame, std::vector<cv::Rect>& boxes, std::vector<float>& scores) {
    boxes.clear(); scores.clear();
    if (frame.empty()) return;
    // Preprocess: YOLOv8 typical ONNX output handled by OpenCV
    if (!net.empty()) {
        try {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640,640), cv::Scalar(), true, false);
            net.setInput(blob);
            std::vector<cv::Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());
            // Parse detections: support common format (N x (5+K))
            for (const auto &out : outs) {
                // each row is [x,y,w,h,conf, class0, class1 ...] OR [batch, anchors, grid, ...]
                if (out.dims == 2 && out.cols >= 6) {
                    for (int r = 0; r < out.rows; ++r) {
                        const float* row = out.ptr<float>(r);
                        float x = row[0], y = row[1], w = row[2], h = row[3];
                        float conf = row[4];
                        if (conf < confThreshold) continue;
                        // find best class
                        int cls = -1; float maxc = 0.f;
                        for (int c = 5; c < out.cols; ++c) {
                            if (row[c] > maxc) { maxc = row[c]; cls = c-5; }
                        }
                        float score = conf * maxc;
                        if (score < confThreshold) continue;
                        // scale to frame size (blob was 640x640)
                        int x0 = static_cast<int>((x - w/2.0f) * frame.cols / 640.0f);
                        int y0 = static_cast<int>((y - h/2.0f) * frame.rows / 640.0f);
                        int bw = static_cast<int>(w * frame.cols / 640.0f);
                        int bh = static_cast<int>(h * frame.rows / 640.0f);
                        boxes.emplace_back(cv::Rect(x0,y0,bw,bh) & cv::Rect(0,0,frame.cols, frame.rows));
                        scores.emplace_back(score);
                    }
                }
                else if (out.dims == 4) {
                    // skip complex outputs
                }
            }
            // apply NMS
            if (!boxes.empty()) {
                std::vector<int> idxs;
                cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, idxs);
                std::vector<cv::Rect> nboxes; std::vector<float> nscores;
                for (int id : idxs) { nboxes.push_back(boxes[id]); nscores.push_back(scores[id]); }
                boxes.swap(nboxes); scores.swap(nscores);
            }
            return;
        } catch(...) {
            std::cerr << "Warning: detector forward failed, falling back to bgsub\n";
        }
    }
    // fallback
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        // Parse detections: support common format (N x (5+K))
        for (const auto &out : outs) {
            // each row is [x,y,w,h,conf, class0, class1 ...] OR [batch, anchors, grid, ...]
            if (out.dims == 2 && out.cols >= 6) {
                for (int r = 0; r < out.rows; ++r) {
                    const float* row = out.ptr<float>(r);
                    float x = row[0], y = row[1], w = row[2], h = row[3];
                    float conf = row[4];
                    if (conf < confThreshold) continue;
                    // find best class
                    int cls = -1; float maxc = 0.f;
                    for (int c = 5; c < out.cols; ++c) {
                        if (row[c] > maxc) { maxc = row[c]; cls = c-5; }
                    }
                    float score = conf * maxc;
                    if (score < confThreshold) continue;
                    // scale to frame size (blob was 640x640)
                    int x0 = static_cast<int>((x - w/2.0f) * frame.cols / 640.0f);
                    int y0 = static_cast<int>((y - h/2.0f) * frame.rows / 640.0f);
                    int bw = static_cast<int>(w * frame.cols / 640.0f);
                    int bh = static_cast<int>(h * frame.rows / 640.0f);
                    boxes.emplace_back(cv::Rect(x0,y0,bw,bh) & cv::Rect(0,0,frame.cols, frame.rows));
                    scores.emplace_back(score);
                }
            }
            else if (out.dims == 4) {
                // skip complex outputs
            }
        }
        // apply NMS
        if (!boxes.empty()) {
            std::vector<int> idxs;
            cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, idxs);
            std::vector<cv::Rect> nboxes; std::vector<float> nscores;
            for (int id : idxs) { nboxes.push_back(boxes[id]); nscores.push_back(scores[id]); }
            boxes.swap(nboxes); scores.swap(nscores);
        }
    } else if (bgSub) {
        cv::Mat fg;
        bgSub->apply(frame, fg);
        cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &c : contours) {
            cv::Rect r = cv::boundingRect(c);
            if (r.area() < 500) continue;
            boxes.push_back(r & cv::Rect(0,0,frame.cols, frame.rows));
            scores.push_back(0.6f);
        }
    }
}

void ObjectTracker::update(const cv::Mat& frame) {
    if (paused.load()) return;
    if (frame.empty()) return;
    // PERFORMANCE: runs at ~12fps
    std::vector<cv::Rect> detBoxes;
    std::vector<float> detScores;
    detect(frame, detBoxes, detScores);

    // Predict existing Kalman tracks
    ktracker.predict();

    // Associate detections to predicted tracks using IOU
    auto predicted = ktracker.getTracks();
    std::vector<bool> detAssigned(detBoxes.size(), false);

    for (const auto &p : predicted) {
        int bestDet = -1; float bestIoU = 0.0f;
        for (size_t d = 0; d < detBoxes.size(); ++d) {
            if (detAssigned[d]) continue;
            float val = iou(p.bbox, detBoxes[d]);
            if (val > bestIoU) { bestIoU = val; bestDet = (int)d; }
        }
        if (bestDet != -1 && bestIoU > 0.3f) {
            ktracker.update(p.id, detBoxes[bestDet]);
            detAssigned[bestDet] = true;
        }
    }

    // Create new tracks for unmatched detections
    for (size_t d = 0; d < detBoxes.size(); ++d) {
        if (!detAssigned[d]) {
            ktracker.createTrack(detBoxes[d]);
        }
    }

    // Erase lost tracks
    ktracker.eraseLostTracks();

    // Build public tracks list from KalmanTracker
    tracks.clear();
    for (const auto &kt : ktracker.getTracks()) {
        Track t;
        t.id = kt.id;
        t.bbox = kt.bbox;
        t.confidence = 1.0f;
        t.velocity = cv::Point2f(kt.kf.statePost.at<float>(2), kt.kf.statePost.at<float>(3));
        t.missed = kt.consecutiveInvisibleCount;
        tracks.push_back(t);
    }
}

void ObjectTracker::draw(cv::Mat& img, const std::vector<int>& selectedIds) {
    for (const auto& t : tracks) {
        cv::Scalar color = (std::find(selectedIds.begin(), selectedIds.end(), t.id) != selectedIds.end()) ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
        cv::rectangle(img, t.bbox, color, 2);
        cv::putText(img, std::to_string(t.id), t.bbox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, color,2);
        // draw velocity
        auto c = cv::Point(t.bbox.x + t.bbox.width/2, t.bbox.y + t.bbox.height/2);
        cv::arrowedLine(img, c, c + cv::Point((int)t.velocity.x*2, (int)t.velocity.y*2), cv::Scalar(0,255,255),2);
        // distance estimate (assume person height for person class)
        if (person_height_m > 0 && t.bbox.height > 0) {
            float Z = focal_px * person_height_m / t.bbox.height; // meters
            std::string dist = std::to_string((int)(Z*100)) + "cm";
            cv::putText(img, dist, cv::Point(t.bbox.x, t.bbox.y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
    }
}

void ObjectTracker::pause() { paused = true; }
void ObjectTracker::resume() { paused = false; }
std::vector<Track> ObjectTracker::getTracks() { return tracks; }

void ObjectTracker::setCameraParams(float f_px, float person_h_m) {
    focal_px = f_px; person_height_m = person_h_m;
}

int ObjectTracker::pickTargetAt(const cv::Point& p) const {
    for (const auto &t : tracks) {
        if (t.bbox.contains(p)) return t.id;
    }
    return -1;
}
