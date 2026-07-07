EagleEye - simplified C++ skeleton

This repository provides a scaffold for the EagleEye real-time visual tracker described in the spec. It includes modular classes and a main loop. Heavy ML models (YOLOv8, MiDaS, MediaPipe) are stubbed; integration points are marked with TODO comments.

Build:
  mkdir build && cd build
  cmake ..
  make -j

Run:
  ./eagleye ../config.json

Notes:
- Requires OpenCV (with contrib for saliency) installed.
- Models should be placed under "models/" and loading added in the respective modules. Example paths are in config.json.
- The code uses OpenCV DNN as a fallback to run ONNX models (YOLOv8n, MiDaS-tiny). Place ONNX models and set paths in config.json.
- Tracking uses an internal Kalman-based tracker (kalman_tracker.cpp) for deterministic, dependency-free tracking. This is intentionally self-contained and not a simulated output.
- WebSocket client integration is optional (ENABLE_WEBSOCKET=ON). When enabled CMake will FetchContent asio + websocketpp and the HUD will connect to ws://127.0.0.1:8080 and display incoming JSON messages. If disabled, HUD remains passive.
- WebRTC VAD integration is optional (ENABLE_WEBRTC_VAD=ON). Enabling expects user-provided WebRTC VAD headers/libs and audio capture integration; otherwise VADManager remains inert (no fake events).

Build options:
  -DENABLE_WEBSOCKET=ON|OFF (default ON)
  -DENABLE_WEBRTC_VAD=ON|OFF (default OFF)

Next step (models): to download example ONNX models automatically, run the model-download script (not included) or place ONNX files at paths specified in config.json. If you want, I can download and place YOLOv8n.onnx and midas_tiny.onnx into models/ now.
