#include "hud.h"

#include <thread>
#include <chrono>
#include <random>

#include "websocket_client.h"
#include <thread>
#include <iostream>
#include <nlohmann/json.hpp>

HUDManager::HUDManager() {
#ifdef ENABLE_WEBSOCKET
    // start websocket client
    std::string uri = "ws://127.0.0.1:8080";
    try {
        auto *ws = new WebSocketClient(uri);
        ws->setMessageCallback([this](const std::string& payload){
            // parse simple JSON message {"sender":"user","content":"..."}
            try {
                auto j = nlohmann::json::parse(payload);
                if (j.contains("content")) pushMessage(j["content"].get<std::string>());
            } catch (...) { pushMessage(payload); }
        });
        if (!ws->start()) {
            delete ws;
        } else {
            // intentionally leaked for simplicity; OS cleans on exit
        }
    } catch (const std::exception &e) { std::cerr << "WS start failed: " << e.what() << std::endl; }
#endif
}

void HUDManager::pushMessage(const std::string& s) {
    std::lock_guard<std::mutex> lk(mtx);
    msgs.push_front(s);
    if (msgs.size() > 5) msgs.pop_back();
}

void HUDManager::draw(cv::Mat& img) {
    std::lock_guard<std::mutex> lk(mtx);
    int y = img.rows - 20;
    for (const auto &m : msgs) {
        cv::putText(img, m, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
        y -= 20;
    }
}
