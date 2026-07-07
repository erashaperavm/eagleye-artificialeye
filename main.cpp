#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include "camera.h"
#include "tracker.h"
#include "pose.h"
#include "optical_flow.h"
#include "saliency.h"
#include "trajectory.h"
#include "vad.h"
#include "hud.h"
#include "tab_scanner.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include "audio.h"

using json = nlohmann::json;

// Globals for mouse callback
static class ObjectTracker* G_TRACKER = nullptr;
static std::vector<int>* G_SELECTED = nullptr;
static bool G_SELECTING = false;
static cv::Point G_SEL_START, G_SEL_END;
static cv::Mat G_LAST_FRAME;

static void onMouse(int event, int x, int y, int flags, void* userdata) {
    (void)flags; (void)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) {
        G_SELECTING = true; G_SEL_START = cv::Point(x,y); G_SEL_END = G_SEL_START;
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (G_SELECTING) G_SEL_END = cv::Point(x,y);
    } else if (event == cv::EVENT_LBUTTONUP) {
        G_SELECTING = false; G_SEL_END = cv::Point(x,y);
        cv::Rect selRect(std::min(G_SEL_START.x, G_SEL_END.x), std::min(G_SEL_START.y, G_SEL_END.y),
                         std::abs(G_SEL_END.x - G_SEL_START.x), std::abs(G_SEL_END.y - G_SEL_START.y));
        if (selRect.width == 0 || selRect.height == 0) {
            // treat as click: pick single
            if (G_TRACKER) {
                int id = G_TRACKER->pickTargetAt(cv::Point(x,y));
                if (id != -1) {
                    G_SELECTED->clear(); G_SELECTED->push_back(id);
                }
            }
        } else {
            // select all tracks intersecting
            if (G_TRACKER) {
                G_SELECTED->clear();
                auto tracks = G_TRACKER->getTracks();
                for (const auto &t : tracks) {
                    if ((selRect & t.bbox).area() > 0) G_SELECTED->push_back(t.id);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Load config
    std::string cfg_path = "config.json";
    if (argc > 1) cfg_path = argv[1];
    json cfg;
    try {
        std::ifstream ifs(cfg_path);
        ifs >> cfg;
    } catch (...) {
        std::cerr << "Failed to read " << cfg_path << "\n";
        return -1;
    }

    CameraCapture camera(cfg["camera"]["index"].get<int>(),
                         cfg["camera"]["width"].get<int>(),
                         cfg["camera"]["height"].get<int>());
    if (!camera.open()) {
        std::cerr << "Camera open failed\n";
        return -1;
    }

    // Resolve model paths with safe fallbacks to repository root filenames if user placed files there
    namespace fs = std::filesystem;
    std::string yoloPath = cfg["models"]["yolo_path"].get<std::string>();
    std::string posePath = cfg["models"]["pose_path"].get<std::string>();
    std::string midasPath = cfg["models"]["midas_path"].get<std::string>();

    auto tryFallback = [&](const std::string &configured, const std::vector<std::string> &candidates){
        if (!configured.empty() && fs::exists(configured)) return configured;
        for (const auto &c : candidates) if (fs::exists(c)) return c;
        return std::string();
    };

    yoloPath = tryFallback(yoloPath, {"./yolov8n.onnx", "yolov8n.onnx", "models/yolov8n.onnx"});
    posePath = tryFallback(posePath, {"./pose.onnx", "pose.onnx", "models/pose.onnx"});
    midasPath = tryFallback(midasPath, {"./midas_tiny.onnx", "midas_tiny.onnx", "models/midas_tiny.onnx"});

    if (yoloPath.empty()) std::cerr << "Warning: YOLO model not found; detection disabled.\n";
    if (posePath.empty()) std::cerr << "Note: pose model not found; pose disabled.\n";
    if (midasPath.empty()) std::cerr << "Note: MiDaS model not found; depth scanner will use fallback.\n";

    auto tracker = std::make_unique<ObjectTracker>(yoloPath);
    auto pose = std::make_unique<PoseEstimator>(posePath);
    auto oflow = std::make_unique<OpticalFlow>();
    auto sal = std::make_unique<SaliencyDetector>();
    auto traj = std::make_unique<TrajectoryPredictor>();
    auto vad = std::make_unique<VADManager>();
    auto hud = std::make_unique<HUDManager>();
    auto scanner = std::make_unique<TabScanner>(midasPath);

    // audio capture
    std::unique_ptr<AudioCapture> audio;
#ifdef ENABLE_PORTAUDIO
    audio = std::make_unique<AudioCapture>();
    if (!audio->start()) {
        audio.reset();
        std::cerr << "Audio capture failed to start; waveform disabled\n";
    }
#else
    (void)0;
#endif

    std::atomic<bool> running(true);
    std::atomic<bool> scanning(false);
    std::atomic<bool> scan_ready(false);
    cv::Mat frozen;

    // Start VAD thread (non-blocking) and connect to HUD
    vad->setCallback([&hud]() {
        hud->pushMessage("Voice detected");
    });
    vad->start();

    const char* window_name = "EagleEye";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    using clock = std::chrono::steady_clock;
    auto last_time = clock::now();
    int64_t frame_count = 0;
    double fps = 0.0;

    // selection state (stored in globals for mouse callback)
    std::vector<int> selectedIds;
    // register mouse callback after selectedIds exists
    G_TRACKER = tracker.get();
    G_SELECTED = &selectedIds;
    cv::setMouseCallback(window_name, onMouse, nullptr);

    // Tab scan thread handle
    std::thread scan_thread;

    while (running.load()) {
        auto t0 = clock::now();
        cv::Mat frame = camera.grabFrame();
        if (frame.empty()) {
            std::cerr << "Empty frame\n";
            break;
        }

        // draw local time on top-right
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char timestr[64];
        std::strftime(timestr, sizeof(timestr), "%F %T", std::localtime(&now_c));

        // Keyboard handling
        int key = cv::waitKey(1);
        if (key == 27) { running = false; break; } // ESC
        if (key == '+' || key == '=') camera.zoomIn();
        if (key == '-') camera.zoomOut();
        bool tabPressed = (key == 9); // Tab key (note: platform dependent)
        // keyboard selection shortcuts: 's' start (center fallback), 'S' cancel
        if (key == 's') {
            G_SELECTING = true; G_SEL_START = cv::Point(frame.cols/2, frame.rows/2); G_SEL_END = G_SEL_START;
        }
        if (key == 'S') { G_SELECTING = false; }

        // mouse selection handled via callback

        if (tabPressed) {
            if (!scanning.load()) {
                // Enter Tab mode (slow mode), don't fully freeze — run at reduced frame rate
                scanning = true;
                scan_ready = false;
                // Launch depth scan thread once
                cv::Mat frameForScan = frame.clone();
                scan_thread = std::thread([&]() {
                    scanner->scan(frameForScan, [&scan_ready, &hud](const cv::Mat& depth){
                        scan_ready = true;
                        hud->pushMessage("Depth scan complete");
                    });
                });
                last_time = clock::now(); // reset timing for slow mode
            }
            // Slow display to ~6 FPS
            static auto last_tab_display = clock::now();
            double tab_interval_ms = 166.0; // ~6fps
            auto nowt = clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(nowt - last_tab_display).count();
            if (dt < tab_interval_ms) {
                // wait a bit to limit update rate
                std::this_thread::sleep_for(std::chrono::milliseconds((int)(tab_interval_ms - dt)));
            }
            last_tab_display = clock::now();

            // compute optical flow on current frame for Tab-mode visualization
            oflow->computeOnFrame(frame);
            // Show current frame with scan effects and depth overlay
            cv::Mat display = frame.clone();
            scanner->drawEffects(display, scan_ready.load());
            oflow->draw(display);
            // draw selection rectangle if any
            if (G_SELECTING) {
                cv::rectangle(display, G_SEL_START, G_SEL_END, cv::Scalar(255,255,255), 2);
            }
            // draw time
            cv::putText(display, timestr, cv::Point(display.cols-250, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
            hud->draw(display);
            cv::imshow(window_name, display);
            // Do not advance trackers while scanning
            frame_count++;
        } else {
            // If previously scanning, stop
            if (scanning.load()) {
                scanning = false;
                // Join thread if running
                if (scan_thread.joinable()) scan_thread.join();
                // Resume modules
                tracker->resume();
                pose->resume();
                oflow->resume();
            }

            // Alternate frame scheduling
            if ((frame_count % 2) == 0) {
                tracker->update(frame); // PERFORMANCE: runs at ~12fps
            } else {
                pose->update(frame);
            }

            // Only compute optical flow when Tab held (user requested: only Tab time)
            sal->update(frame);
            traj->update(tracker->getTracks());

            // HUD draws boxes, pose, arrows, heatmap, messages
            cv::Mat display = frame.clone();
            tracker->draw(display, selectedIds);
            pose->draw(display);
            // optical flow not drawn normally
            sal->draw(display);
            traj->draw(display, selectedIds);
            // draw selection rectangle overlay if user is selecting
            if (G_SELECTING) cv::rectangle(display, G_SEL_START, G_SEL_END, cv::Scalar(255,255,255), 2);
            // draw time
            cv::putText(display, timestr, cv::Point(display.cols-250, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
            hud->draw(display);
            // draw audio heatmap (right-bottom) if available
#ifdef ENABLE_PORTAUDIO
            if (audio) {
                // maintain angular bins for heatmap
                static std::vector<float> bins(36, 0.0f);
                float dir = audio->getDirection();
                float level = audio->getLevel();
                if (!std::isnan(dir)) {
                    int idx = (int)((dir*0.5f + 0.5f) * (bins.size()-1));
                    if (idx < 0) idx = 0; if (idx >= (int)bins.size()) idx = bins.size()-1;
                    bins[idx] += level * 5.0f; // accumulate
                }
                // decay
                for (auto &b : bins) b *= 0.96f;
                int w = 160, h = 160;
                cv::Mat heat(h, w, CV_8UC3, cv::Scalar(0,0,0));
                // draw radial heatmap
                int cx = w/2, cy = h/2, R = std::min(w,h)/2 - 4;
                for (int a = 0; a < (int)bins.size(); ++a) {
                    float v = std::min(1.0f, bins[a]);
                    float angle = (float)a / bins.size() * 2.0f * (float)CV_PI;
                    int x = cx + (int)(std::cos(angle) * (R* (0.3f + v*0.7f)));
                    int y = cy + (int)(std::sin(angle) * (R* (0.3f + v*0.7f)));
                    cv::Scalar col = cv::Scalar(0, 255 * v, 255 * (1.0 - v));
                    cv::circle(heat, cv::Point(x,y), std::max(2, (int)(v*8)), col, -1);
                }
                // overlay heat onto display bottom-right
                cv::Mat roi = display(cv::Rect(display.cols - w - 10, display.rows - h - 10, w, h));
                cv::addWeighted(heat, 0.7, roi, 0.3, 0, roi);
            }
#endif

            cv::imshow(window_name, display);
            frame_count++;
        }

        // FPS calc
        auto t1 = clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - last_time).count();
        if (elapsed >= 250) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            last_time = t1;
            std::string title = std::string("EagleEye - FPS: ") + std::to_string((int)fps);
            cv::setWindowTitle(window_name, title);
        }
    }

    // Cleanup
    running = false;
    if (scan_thread.joinable()) scan_thread.join();
    vad->stop();

    return 0;
}
