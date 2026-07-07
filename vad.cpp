#include "vad.h"
#include <chrono>
#include <iostream>

VADManager::VADManager() : running(false) {}
VADManager::~VADManager() { stop(); }

void VADManager::start() {
    // NO-OP by default. To enable real VAD, compile/link with WebRTC VAD and implement audio capture.
}
void VADManager::stop() {
    // NO-OP
}
void VADManager::setCallback(std::function<void()> c) { cb = c; }
