cmake_minimum_required(VERSION 3.15)
project(eagleye_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui imgcodecs videoio video dnn)
# Check for saliency header (contrib). If available, enable saliency module.
include(CheckIncludeFileCXX)
check_include_file_cxx("opencv2/saliency.hpp" HAVE_OPENCV_SALIENCY)
if(HAVE_OPENCV_SALIENCY)
    add_definitions(-DHAVE_OPENCV_SALIENCY=1)
endif()
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

option(ENABLE_WEBSOCKET "Enable building websocket client" ON)
option(ENABLE_WEBRTC_VAD "Enable building with WebRTC VAD (user must provide lib)" OFF)

set(SRC_FILES
    main.cpp
    camera.cpp
    tracker.cpp
    kalman_tracker.cpp
    pose.cpp
    optical_flow.cpp
    optical_flow_tab.cpp
    trajectory.cpp
    vad.cpp
    hud.cpp
    tab_scanner.cpp
    websocket_client.cpp
    audio.cpp
)

if(HAVE_OPENCV_SALIENCY)
    list(APPEND SRC_FILES saliency.cpp)
else()
    message(STATUS "OpenCV saliency headers not found; building without saliency support (using stub)")
    list(APPEND SRC_FILES saliency_stub.cpp)
endif()

add_executable(eagleye ${SRC_FILES})

# find PortAudio and link after target created
find_path(PORTAUDIO_INCLUDE portaudio.h)
find_library(PORTAUDIO_LIB portaudio)
if(PORTAUDIO_INCLUDE AND PORTAUDIO_LIB)
    message(STATUS "Found PortAudio for audio capture")
    target_include_directories(eagleye PRIVATE ${PORTAUDIO_INCLUDE})
    target_link_libraries(eagleye PRIVATE ${PORTAUDIO_LIB})
    target_compile_definitions(eagleye PRIVATE ENABLE_PORTAUDIO=1)
else()
    message(STATUS "PortAudio not found: audio capture disabled; waveform will be empty")
endif()

# If enabled, pull websocketpp and asio via FetchContent
include(FetchContent)
if(ENABLE_WEBSOCKET)
    message(STATUS "ENABLE_WEBSOCKET=ON: Fetching asio and websocketpp via FetchContent")
    FetchContent_Declare(
      asio
      GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
      GIT_TAG asio-1-28-1
    )
    FetchContent_Declare(
      websocketpp
      GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
      GIT_TAG 0.8.2
    )
    FetchContent_MakeAvailable(asio websocketpp)
    target_compile_definitions(eagleye PRIVATE ENABLE_WEBSOCKET=1 ASIO_STANDALONE=1)
    target_include_directories(eagleye PRIVATE ${asio_SOURCE_DIR}/asio/include ${websocketpp_SOURCE_DIR})
else()
    message(STATUS "ENABLE_WEBSOCKET=OFF: websocket client disabled")
endif()

if(ENABLE_WEBRTC_VAD)
    message(STATUS "ENABLE_WEBRTC_VAD=ON: expecting system-provided WebRTC VAD library and headers")
    target_compile_definitions(eagleye PRIVATE ENABLE_WEBRTC_VAD=1)
endif()

target_include_directories(eagleye PRIVATE ${OpenCV_INCLUDE_DIRS})

# link PortAudio if found is handled above

target_link_libraries(eagleye PRIVATE ${OpenCV_LIBS} Threads::Threads nlohmann_json::nlohmann_json)

# PERFORMANCE: add compile flags for optimization
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Recommended: compile with O3 for performance
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(eagleye PRIVATE -O3 -march=native -funroll-loops)
endif()
