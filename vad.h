#pragma once
#include <atomic>
#include <functional>
#include <thread>

class VADManager {
public:
    VADManager();
    ~VADManager();
    void start();
    void stop();
    void setCallback(std::function<void()> cb);
private:
    std::atomic<bool> running;
    std::thread worker;
    std::function<void()> cb;
};
