#pragma once
#include <functional>
#include <string>

class WebSocketClient {
public:
    WebSocketClient(const std::string& uri);
    ~WebSocketClient();
    bool start();
    void stop();
    void setMessageCallback(std::function<void(const std::string&)> cb);
private:
    struct Impl;
    Impl* impl;
};
