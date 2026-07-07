#pragma once
#include <vector>
#include <atomic>

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    bool start();
    void stop();
    // get latest waveform samples (normalized -1..1)
    void getWaveform(std::vector<float>& out);
    // get simple direction estimate: -1..1 (left..right), NAN if unknown
    float getDirection();
    // VAD detection by energy
    bool getVAD();
    // last RMS energy
    float getLevel();
private:
    struct Impl;
    Impl* impl;
};
