#include "websocket_client.h"
#include <iostream>
#include <thread>
#include <atomic>

#ifdef ENABLE_WEBSOCKET
  #define ASIO_STANDALONE
  #include <websocketpp/config/asio_no_tls_client.hpp>
  #include <websocketpp/client.hpp>
#endif

// This implementation uses websocketpp + asio (standalone) when available.
// If the libraries are not present at compile time, the Impl will be functional but start() will no-op.

struct WebSocketClient::Impl {
    std::string uri;
    std::function<void(const std::string&)> cb;
    std::atomic<bool> running{false};
    std::thread th;
};

WebSocketClient::WebSocketClient(const std::string& uri) {
    impl = new Impl();
    impl->uri = uri;
}
WebSocketClient::~WebSocketClient() {
    stop();
    delete impl;
}

bool WebSocketClient::start() {
#ifdef ENABLE_WEBSOCKET
    if (!impl) return false;
    if (impl->running) return true;
    impl->running = true;
    impl->th = std::thread([this]() {
        try {
            typedef websocketpp::client<websocketpp::config::asio_client> client;
            client c;
            c.init_asio();
            c.clear_access_channels(websocketpp::log::alevel::all);
            c.clear_error_channels(websocketpp::log::elevel::all);

            websocketpp::lib::error_code ec;
            auto con = c.get_connection(impl->uri, ec);
            if (ec) {
                std::cerr << "WebSocket connect error: " << ec.message() << std::endl;
                impl->running = false; return;
            }
            con->set_message_handler([this](websocketpp::connection_hdl, client::message_ptr msg){
                if (impl->cb) impl->cb(msg->get_payload());
            });
            c.connect(con);
            c.run();
        } catch (const std::exception &e) {
            std::cerr << "WebSocket client exception: " << e.what() << std::endl;
        }
        impl->running = false;
    });
    return true;
#else
    (void)impl; // no-op
    std::cerr << "WebSocket support not enabled at compile time." << std::endl;
    return false;
#endif
}

void WebSocketClient::stop() {
    if (!impl) return;
    if (impl->running) {
#ifdef ENABLE_WEBSOCKET
        // websocketpp run loop stops when connection closed; we don't have handle to stop cleanly here
#endif
        if (impl->th.joinable()) impl->th.join();
    }
}

void WebSocketClient::setMessageCallback(std::function<void(const std::string&)> cb) {
    if (!impl) return; impl->cb = cb;
}
